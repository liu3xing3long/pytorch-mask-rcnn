"""
Mask R-CNN
Configurations and data loading code for MS COCO.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 coco.py train --dataset=/path/to/coco/ --model=coco

    # Train a new model starting from ImageNet weights
    python3 coco.py train --dataset=/path/to/coco/ --model=imagenet

    # Continue training a model that you had trained earlier
    python3 coco.py train --dataset=/path/to/coco/ --model=/path/to/weights.h5

    # Continue training the last model you trained
    python3 coco.py train --dataset=/path/to/coco/ --model=last

    # Run COCO evaluatoin on the last model you trained
    python3 coco.py evaluate --dataset=/path/to/coco/ --model=last
"""

import os
from lib import model as modellib
from lib.config import CocoConfig
from datasets.coco import CocoDataset
from tools.utils import evaluate_coco
# Root directory of the project
DEFAULT_DATASET_PATH = os.path.join('/home/hongyang/dataset/coco')
DEFAULT_LOGS_DIR = os.path.join(os.getcwd(), "logs")
DEFAULT_DATASET_YEAR = "2014"


if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN on MS COCO.')
    parser.add_argument('--phase',
                        required=False,
                        default='train',
                        help='train or evaluate')
    parser.add_argument('--config',
                        required=False,
                        default='hyli_default',
                        metavar='config name')
    parser.add_argument('--model',
                        default='last',
                        metavar="/path/to/weights.pth",
                        help="Path to weights .pth file or [coco/imagenet/last]")

    parser.add_argument('--device_id',
                        default='0,1,2', type=str)
    parser.add_argument('--dataset_path',
                        default=DEFAULT_DATASET_PATH,
                        metavar="/path/to/coco/",
                        help='Directory of the MS-COCO dataset')
    parser.add_argument('--logs',
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
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
    print("Logs: ", args.logs)
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
    model = modellib.MaskRCNN(config=config, model_dir=args.logs)
    model = model.cuda()
    #TODO
    # model = torch.nn.DataParallel(model).cuda()

    # Select weights file to load
    if args.model:
        if args.model.lower() == "coco_pretrain":
            model_path = config.COCO_PRETRAIN_MODEL_PATH
            suffix = 'coco pretrain'
        elif args.model.lower() == "imagenet_pretrain":
            model_path = config.IMAGENET_PRETRAIN_MODEL_PATH
            suffix = 'imagenet pretrain'
        elif args.model.lower() == "last":
            # Find last trained weights
            model_path = model.find_last()[1]
            suffix = 'last trained model for resume or eval'
        else:
            model_path = args.model
            suffix = 'designated'
    else:
        model_path = ""
        suffix = 'empty!!! train from scratch!!!'

    # Load weights
    print("Loading weights ({:s})\t{:s}".format(suffix, model_path))
    model.load_weights(model_path)

    # Train or evaluate
    if args.phase == "train":
        # Training dataset. Use the training set and 35K from the
        # validation set, as as in the Mask RCNN paper.
        dataset_train = CocoDataset()
        dataset_train.load_coco(args.dataset_path, "train", year=args.year, auto_download=args.download)
        dataset_train.load_coco(args.dataset_path, "valminusminival", year=args.year, auto_download=args.download)
        dataset_train.prepare()

        # Validation dataset
        dataset_val = CocoDataset()
        dataset_val.load_coco(args.dataset_path, "minival", year=args.year, auto_download=args.download)
        dataset_val.prepare()

        # *** This training schedule is an example. Update to your needs ***
        # Training - Stage 1
        print("Training network heads")
        model.train_model(dataset_train, dataset_val,
                          learning_rate=config.LEARNING_RATE,
                          epochs=40,
                          layers='heads')

        # Training - Stage 2
        # Finetune layers from ResNet stage 4 and up
        print("Fine tune Resnet stage 4 and up")
        model.train_model(dataset_train, dataset_val,
                          learning_rate=config.LEARNING_RATE,
                          epochs=120,
                          layers='4+')

        # Training - Stage 3
        # Fine tune all layers
        print("Fine tune all layers")
        model.train_model(dataset_train, dataset_val,
                          learning_rate=config.LEARNING_RATE / 10,
                          epochs=160,
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
