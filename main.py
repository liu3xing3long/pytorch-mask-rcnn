import argparse
import lib.network as network
from lib.config import CocoConfig
from lib.workflow import *
from tools.utils import update_config_and_load_model

if __name__ == '__main__':
    # weird: if put ahead; import error occurs
    from datasets.dataset_coco import get_data

    parser = argparse.ArgumentParser(description='Mask R-CNN')

    parser.add_argument('--phase',
                        # default='train',
                        default='inference',
                        help='train or inference')

    parser.add_argument('--config_name',
                        required=False,
                        # default='all_new_2')
                        default='hyli_default_old')

    parser.add_argument('--debug',
                        default=1, type=int)  # no bool type here please

    parser.add_argument('--device_id',
                        default='0,1', type=str)

    args = parser.parse_args()
    print('\nSTART::: phase is [{:s}]'.format(args.phase.upper()))

    # Configuration
    config = CocoConfig(args)

    # Create model
    print('building network ...')
    model = network.MaskRCNN(config)

    # Select weights file to load
    # update start epoch and iter if resume
    config, model = update_config_and_load_model(config, model)

    if config.MISC.GPU_COUNT < 1:
        print('CPU mode ...')
    elif config.MISC.GPU_COUNT == 1:
        print('single gpu mode ...')
        model = model.cuda()
    else:
        model = torch.nn.DataParallel(model).cuda()

    # Get data
    train_data, val_data, val_api = get_data(config)

    # Train or inference
    if args.phase == 'train':
        # TODO: training workflow in accordance with Detectron
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

    elif args.phase == 'inference':

        test_model(model, val_data, val_api)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'evaluate'".format(args.phase))
