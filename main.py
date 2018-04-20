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
                        default='evaluate',
                        help='train or evaluate')
    parser.add_argument('--config_name',
                        required=False,
                        default='all_new')
                        # default='hyli_default_old')
    parser.add_argument('--debug',
                        default=1, type=int)
    parser.add_argument('--device_id',
                        default='0,1', type=str)
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
        config = CocoConfig(config_name=args.config_name, args=args)
    elif args.phase == 'evaluate':
        class InferenceConfig(CocoConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            DETECTION_MIN_CONFIDENCE = 0
        config = InferenceConfig(config_name=args.config_name, args=args)

    # Create model
    print('building network ...')
    model = network.MaskRCNN(config, model_dir=args.results)
    # Select weights file to load
    config = select_weights(args, config, model)
    config.display(config.LOG_FILE)
    model.config = config

    model = model.cuda()
    if config.GPU_COUNT > 1:
        model = torch.nn.DataParallel(model).cuda()

    train_data, val_data, val_api = get_data(config, args)

    # Train or evaluate
    if args.phase == 'train':
        # TODO (low): to consider inference during training
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

    elif args.phase == 'evaluate':

        test_model(model, val_data, val_api)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'evaluate'".format(args.phase))
