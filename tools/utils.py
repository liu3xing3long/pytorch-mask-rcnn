import os
import torch
from torch.autograd import Variable
# import cv2


############################################################
#  Pytorch Utility Functions
############################################################
def unique1d(variable):
    variable = variable.squeeze()
    assert variable.dim() == 1
    if variable.size(0) == 1:
        return variable
    variable = variable.sort()[0]
    unique_bool = variable[1:] != variable[:-1]
    first_element = Variable(torch.ByteTensor([True]), requires_grad=False)
    if variable.is_cuda:
        first_element = first_element.cuda()
    unique_bool = torch.cat((first_element, unique_bool), dim=0)
    return variable[unique_bool]


def intersect1d(variable1, variable2):
    aux = torch.cat((variable1, variable2), dim=0)
    aux = aux.squeeze().sort()[0]
    return aux[:-1][(aux[1:] == aux[:-1])]


def log2(x):
    """Implementation of Log2. Pytorch doesn't have a native implementation."""
    ln2 = Variable(torch.log(torch.FloatTensor([2.0])), requires_grad=False)
    if x.is_cuda:
        ln2 = ln2.cuda()
    return torch.log(x) / ln2

############################################################
#  Logging Utility Functions
############################################################
# def log(text, array=None):
#     """Prints a text message. And, optionally, if a Numpy array is provided it
#     prints it's shape, min, and max values.
#     """
#     if array is not None:
#         text = text.ljust(25)
#         text += ("shape: {:20}  min: {:10.5f}  max: {:10.5f}".format(
#             str(array.shape),
#             array.min() if array.size else "",
#             array.max() if array.size else ""))
#     print(text)
#
#
# def printProgressBar(iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
#     """
#     Call in a loop to create terminal progress bar
#     @params:
#         iteration   - Required  : current iteration (Int)
#         total       - Required  : total iterations (Int)
#         prefix      - Optional  : prefix string (Str)
#         suffix      - Optional  : suffix string (Str)
#         decimals    - Optional  : positive number of decimals in percent complete (Int)
#         length      - Optional  : character length of bar (Int)
#         fill        - Optional  : bar fill character (Str)
#     """
#     percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
#     filledLength = int(length * iteration // total)
#     bar = fill * filledLength + '-' * (length - filledLength)
#     print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\n')
#     # Print New Line on Complete
#     if iteration == total:
#         print()


def remove(file_name):
    try:
        os.remove(file_name)
    except:
        pass


def print_log(msg, file=None, init=False):

    print(msg)
    if file is None:
        pass
    else:
        if init:
            remove(file)
        with open(file, 'a') as log_file:
            log_file.write('%s\n' % msg)


def _find_last(config):

    dir_name = os.path.join('results', config.CTRL.CONFIG_NAME.lower(), 'train')
    # Find the last checkpoint
    checkpoints = next(os.walk(dir_name))[2]
    checkpoints = filter(lambda f: f.startswith("mask_rcnn"), checkpoints)
    checkpoints = sorted(checkpoints)
    if not checkpoints:
        return dir_name, None
    checkpoint = os.path.join(dir_name, checkpoints[-1])
    return dir_name, checkpoint


def update_config_and_load_model(config, network):

    choice = config.MODEL.INIT_FILE_CHOICE
    phase = config.CTRL.PHASE

    if phase == 'train':

        if os.path.exists(choice):
            print('[{:s}]loading designated weights\t{:s}\n'.format(phase.upper(), choice))
            model_path = choice
            del config.MODEL.PRETRAIN_COCO_MODEL_PATH
            del config.MODEL.PRETRAIN_IMAGENET_MODEL_PATH
        else:
            model_path = _find_last(config)[1]
            if model_path is not None:
                if choice.lower() in ['coco_pretrain', 'imagenet_pretrain']:
                    print('WARNING: find existing model... ignore pretrain model')
                    del config.MODEL.PRETRAIN_COCO_MODEL_PATH
                    del config.MODEL.PRETRAIN_IMAGENET_MODEL_PATH
            else:
                if choice.lower() == "imagenet_pretrain":
                    model_path = config.MODEL.PRETRAIN_IMAGENET_MODEL_PATH
                    suffix = 'imagenet'
                    del config.MODEL.PRETRAIN_COCO_MODEL_PATH
                elif choice.lower() == "coco_pretrain":
                    model_path = config.MODEL.PRETRAIN_COCO_MODEL_PATH
                    suffix = 'coco'
                    del config.MODEL.PRETRAIN_IMAGENET_MODEL_PATH
                print('use {:s} pretrain model...'.format(suffix))

        print('loading weights \t{:s}\n'.format(model_path))

    elif phase == 'inference':

        del config.MODEL.PRETRAIN_COCO_MODEL_PATH
        del config.MODEL.PRETRAIN_IMAGENET_MODEL_PATH

        if choice.lower() in ['coco_pretrain', 'imagenet_pretrain', 'last']:
            model_path = _find_last(config)[1]
            print('use last trained model for inference')
        elif os.path.exists(choice):
            model_path = choice
            print('use designated model for inference')
        print('[{:s}] loading model weights\t{:s} for inference\n'.format(phase.upper(), model_path))

    checkpoints = torch.load(model_path)
    network.load_state_dict(checkpoints['state_dict'])

    if phase == 'train':
        network.start_epoch = checkpoints['epoch']
        network.start_iter = checkpoints['iter']
        # init counters
        network.epoch = network.start_epoch
        network.iter = network.start_iter

    # add new info to config
    config.MODEL.INIT_MODEL = model_path

    if phase == 'train':
        config.MISC.LOG_FILE = os.path.join(
            config.MISC.RESULT_FOLDER,
            'train_log_start_ep_{:04d}_iter_{:04d}.txt'.format(network.start_epoch, network.start_iter))
    else:
        model_name = os.path.basename(model_path).replace('.pth', '')
        config.MISC.LOG_FILE = os.path.join(
            config.MISC.RESULT_FOLDER, 'inference_from_{:s}.txt'.format(model_name))
        model_suffix = os.path.basename(model_path).replace('mask_rcnn_', '')

        config.MISC.DET_RESULT_FILE = os.path.join(config.MISC.RESULT_FOLDER, 'det_result_{:s}'.format(model_suffix))
        config.MISC.SAVE_IMAGE_DIR = os.path.join(config.MISC.RESULT_FOLDER, model_suffix.replace('.pth', ''))
        if not os.path.exists(config.MISC.SAVE_IMAGE_DIR):
            os.makedirs(config.MISC.SAVE_IMAGE_DIR)

    config.display(config.MISC.LOG_FILE)
    network.config = config

    return config, network

