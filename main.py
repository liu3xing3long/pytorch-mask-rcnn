"""
Mask R-CNN
Configurations and data loading code for MS COCO.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

"""
import argparse
import lib.network as network
from lib.config import CocoConfig
from lib.model import *

# Root directory of the project
DEFAULT_DATASET_PATH = os.path.join(os.getcwd(), 'datasets/coco')
DEFAULT_LOGS_DIR = os.path.join(os.getcwd(), 'results')
DEFAULT_DATASET_YEAR = '2014'


if __name__ == '__main__':
    # weird: if put ahead; import error occurs
    from datasets.dataset_coco import CocoDataset, get_data

    parser = argparse.ArgumentParser(description='Train Mask R-CNN on MS COCO.')
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
                        help="Path to weights .pth file or coco/imagenet/last")
    parser.add_argument('--debug',
                        default=True, type=bool)

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
    print('\nSTART::: phase is [{:s}]'.format(args.phase))

    # Configuration
    config = []
    if args.phase == "train":
        config = CocoConfig(config_name=args.config, args=args)
    elif args.phase == 'evaluate':
        class InferenceConfig(CocoConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            DETECTION_MIN_CONFIDENCE = 0
        config = InferenceConfig(config_name=args.config, args=args)

    # Create model
    print('building network ...')
    model = network.MaskRCNN(config, model_dir=args.results)
    # Select weights file to load
    config = select_weights(args, config, model)
    config.display(config.LOG_FILE)
    model.config = config

    model = model.cuda()
    # TODO: multi-gpu
    # model = torch.nn.DataParallel(model).cuda()

    # Train or evaluate
    if args.phase == 'train':

        train_data, val_data = get_data(config, args)
        # *** This training schedule is an example. Update to your needs ***
        # Training - Stage 1
        print("\nTraining network heads")
        train_model(model, train_data, val_data,
                    lr=config.LEARNING_RATE, total_ep_curr_call=40, layers='heads')

        # Training - Stage 2
        # Finetune layers from ResNet stage 4 and up
        print("\nFinetune Resnet stage 4 and up")
        train_model(model, train_data, val_data,
                    lr=config.LEARNING_RATE, total_ep_curr_call=120, layers='4+')

        # Training - Stage 3
        # Fine tune all layers
        print("\nFine tune all layers")
        train_model(model, train_data, val_data,
                    lr=config.LEARNING_RATE / 10, total_ep_curr_call=160, layers='all')

    elif args.phase == "evaluate":
        # Validation dataset
        dataset_val = CocoDataset()
        dataset_val.prepare()
        val_num = dataset_val.num_images if args.limit == -1 else args.limit
        print("Running COCO evaluation on {} images.".format(val_num))
        coco_api = dataset_val.load_coco(args.dataset_path, "minival", year=args.year,
                                         return_coco_api=True, auto_download=args.download)
        evaluate_coco(model, dataset_val, coco_api, "bbox", limit=int(args.limit))
        # evaluate_coco(model, dataset_val, coco, "segm", limit=int(args.limit))
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'evaluate'".format(args.phase))
