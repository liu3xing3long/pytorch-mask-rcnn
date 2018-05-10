import argparse
import lib.model as network
from lib.config import CocoConfig
from lib.workflow import *
from tools.utils import update_config_and_load_model, set_optimizer, check_max_mem
from datasets.dataset_coco import get_data
from tools.visualize import Visualizer

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Mask R-CNN')

    parser.add_argument('--phase',
                        default='train',
                        # default='inference',
                        help='train or inference')

    parser.add_argument('--config_name',
                        required=False,
                        # default='')
                        default='local_pc')
                        # default='base_101_quick')

    parser.add_argument('--config_file',
                        default=None)
                        # default='configs/meta_101_quick_3_l1_sig_multi.yaml')
                        # default='configs/meta_101_quick_3.yaml')

    # debug mode: set train_data to val_data for faster data loading.
    # show loss step by step; smaller input image size
    # do validation right after a few steps and visualize predictions
    parser.add_argument('--debug',
                        default=1, type=int)  # no bool type here please

    parser.add_argument('--device_id',
                        default='0,1', type=str)

    parser.add_argument('opts',
                        help='See lib/config.py for all options',
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    print('\nSTART::: phase is [{:s}]'.format(args.phase.upper()))

    # Configuration
    config = CocoConfig(args)
    # Get data
    train_data, val_data, val_api = get_data(config)

    # Create model
    print('building network ...\n')
    model = network.MaskRCNN(config)
    if config.MISC.GPU_COUNT < 1:
        print('cpu mode ...')
    elif config.MISC.GPU_COUNT == 1:
        print('single gpu mode ...')
        model = model.cuda()
    else:
        print('multi-gpu mode ...')
        model = torch.nn.DataParallel(model).cuda()

    if args.phase == 'train':
        optimizer = set_optimizer(model, config.TRAIN) if config.CTRL.DEBUG \
            else check_max_mem(model, train_data)

    # Select weights file to load (MUST be put at the end)
    # update start epoch and iter if resume
    config, model = update_config_and_load_model(config, model, train_data)

    # Visualizer
    vis = Visualizer(config, model, val_data)

    print_log('print network structure in log file [NOT shown in terminal] ...', config.MISC.LOG_FILE)
    print_log(model, config.MISC.LOG_FILE, quiet_termi=True)

    # Train or inference
    if args.phase == 'train':

        # Training - Stage 1
        print("\nTraining network heads")
        train_model(model, train_data, val_data,
                    optimizer=optimizer, layers='heads', coco_api=val_api, vis=vis)

        # Training - Stage 2
        # Finetune layers from ResNet stage 4 and up
        print("\nFinetune Resnet stage 4 and up")
        train_model(model, train_data, val_data,
                    optimizer=optimizer, layers='4+', coco_api=val_api, vis=vis)

        # Training - Stage 3
        # Fine tune all layers
        print("\nFine tune all layers")
        train_model(model, train_data, val_data,
                    optimizer=optimizer, layers='all', coco_api=val_api, vis=vis)

    elif args.phase == 'inference':

        test_model(model, val_data, val_api, during_train=False, vis=vis)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'evaluate'".format(args.phase))
