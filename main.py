"""
Mask R-CNN
Configurations and data loading code for MS COCO.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

"""
import argparse
import torch
import torch.utils.data
from tools.utils import evaluate_coco
import lib.network as network
from lib.config import CocoConfig
from lib.model import *

# Root directory of the project
DEFAULT_DATASET_PATH = os.path.join(os.getcwd(), 'datasets/coco')
DEFAULT_LOGS_DIR = os.path.join(os.getcwd(), 'results')
DEFAULT_DATASET_YEAR = '2014'


if __name__ == '__main__':
    # weird: if put ahead; import error occurs
    from datasets.dataset_coco import CocoDataset, DatasetPack
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN on MS COCO.')
    parser.add_argument('--phase',
                        required=False,
                        default='train',
                        help='train or evaluate')
    parser.add_argument('--config',
                        required=False,
                        default='hyli_default_old',
                        metavar='config name')
    parser.add_argument('--model',
                        default='last',
                        metavar="/path/to/weights.pth",
                        help="Path to weights .pth file or [coco/imagenet/last]")

    parser.add_argument('--device_id',
                        default='0', type=str)
    parser.add_argument('--dataset_path',
                        default=DEFAULT_DATASET_PATH,
                        metavar="/path/to/coco/",
                        help='Directory of the MS-COCO dataset')
    parser.add_argument('--results',
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/results/",
                        help='Logs and checkpoints directory (default=results/)')
    parser.add_argument('--year',
                        default=DEFAULT_DATASET_YEAR,
                        metavar="<year>",
                        help='Year of the MS-COCO dataset (2014 or 2017) (default=2014)')
    parser.add_argument('--limit',
                        default=-1,
                        metavar="<image count>",
                        help='Images to use for evaluation, -1 means all val images')
    parser.add_argument('--download',
                        default=False,
                        metavar="<True|False>",
                        help='Automatically download and unzip MS-COCO files',
                        type=bool)

    args = parser.parse_args()
    print("Phase: ", args.phase)
    print("Model: ", args.model)
    print("Dataset: ", args.dataset_path)
    print("Year: ", args.year)
    print("Logs: ", args.results)
    print("Auto download: ", args.download)
    print("Config name: ", args.config)

    # Configurations
    if args.phase == "train":
        config = CocoConfig(config_name=args.config, args=[args.device_id])
    elif args.phase == 'evaluate':
        class InferenceConfig(CocoConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            DETECTION_MIN_CONFIDENCE = 0
        config = InferenceConfig(config_name=args.config)
    config.display()

    # Create model
    print('building network ...')
    network = network.MaskRCNN(config, args.results)
    # Select weights file to load
    if args.model:
        if args.model.lower() == "coco_pretrain":
            model_path = config.PRETRAIN_COCO_MODEL_PATH
            suffix = 'coco pretrain'
        elif args.model.lower() == "imagenet_pretrain":
            model_path = config.PRETRAIN_IMAGENET_MODEL_PATH
            suffix = 'imagenet pretrain'
        elif args.model.lower() == "last":
            # Find last trained weights
            model_path = find_last(config, args.results)[1]
            suffix = 'last trained model for resume or eval'
        else:
            model_path = args.model
            suffix = 'designated'
    else:
        model_path = ""
        suffix = 'empty!!! train from scratch!!!'
    print("Loading weights ({:s})\t{:s}\n".format(suffix, model_path))
    network.load_weights(model_path)
    model = network.cuda()
    # TODO: multi-gpu
    # model = torch.nn.DataParallel(model).cuda()

    # Train or evaluate
    if args.phase == "train":
        # train data
        dset_train = CocoDataset()
        # TODO: for debug, skip loading training data
        print('load train data...')
        dset_train.load_coco(args.dataset_path, "train", year=args.year, auto_download=args.download)
        print('load val_minus_minival data...')
        dset_train.load_coco(args.dataset_path, "valminusminival", year=args.year, auto_download=args.download)
        dset_train.prepare()
        # validation data
        dset_val = CocoDataset()
        print('load minival data...')
        dset_val.load_coco(args.dataset_path, "minival", year=args.year, auto_download=args.download)
        dset_val.prepare()

        # Data generators
        train_set = DatasetPack(dset_train, config, augment=True)
        train_generator = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=True, num_workers=4)
        val_set = DatasetPack(dset_val, config, augment=True)
        val_generator = torch.utils.data.DataLoader(val_set, batch_size=1, shuffle=True, num_workers=4)

        # *** This training schedule is an example. Update to your needs ***
        # Training - Stage 1
        print("\nTraining network heads")
        train_model(model, train_generator, val_generator,
                    lr=config.LEARNING_RATE,
                    total_ep_curr_call=40,
                    layers='heads')

        # Training - Stage 2
        # Finetune layers from ResNet stage 4 and up
        print("\nFine tune Resnet stage 4 and up")
        train_model(model, train_generator, val_generator,
                    lr=config.LEARNING_RATE,
                    total_ep_curr_call=120,
                    layers='4+')

        # Training - Stage 3
        # Fine tune all layers
        print("\nFine tune all layers")
        train_model(model, train_generator, val_generator,
                    lr=config.LEARNING_RATE / 10,
                    total_ep_curr_call=160,
                    layers='all')

    elif args.phase == "evaluate":
        # Validation dataset
        dataset_val = CocoDataset()
        coco_api = dataset_val.load_coco(args.dataset_path, "minival", year=args.year,
                                         return_coco_api=True, auto_download=args.download)
        dataset_val.prepare()
        print("Running COCO evaluation on {} images.".format(args.limit))
        evaluate_coco(model, dataset_val, coco_api, "bbox", limit=int(args.limit))
        # evaluate_coco(model, dataset_val, coco, "segm", limit=int(args.limit))
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'evaluate'".format(args.phase))
