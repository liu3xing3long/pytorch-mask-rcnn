import torch
from torch.autograd import Variable
import torch.optim as optim
import os
import math
from tools.collections import AttrDict
import yaml
import copy
from past.builtins import basestring
import numpy as np
from ast import literal_eval


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
# def printProgressBar(iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = ''):
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


def print_log(msg, file=None, init=False, additional_file=None):

    print(msg)
    if file is None:
        pass
    else:
        if init:
            remove(file)
        with open(file, 'a') as log_file:
            log_file.write('%s\n' % msg)

        if additional_file is not None:
            # TODO (low): a little buggy here: no removal of previous additional_file
            with open(additional_file, 'a') as addition_log:
                addition_log.write('%s\n' % msg)


def compute_left_time(iter_avg, curr_ep, total_ep, curr_iter, total_iter):

    total_time = ((total_iter - curr_iter) + (total_ep - curr_ep)*total_iter) * iter_avg
    days = math.floor(total_time / (3600*24))
    hrs = (total_time - days*3600*24) / 3600
    return days, hrs


def _cls2dict(config):
    output = AttrDict()
    for a in dir(config):
        value = getattr(config, a)
        if not a.startswith("__") and not callable(value):
            assert isinstance(value, AttrDict)
            output[a] = value
    return output


def _dict2cls(_config, config):
    for a in dir(config):
        if not a.startswith("__") and not callable(getattr(config, a)):
            setattr(config, a, _config[a])


def merge_cfg_from_file(cfg_filename, config):
    """Load a yaml config file and merge it into the global config."""
    with open(cfg_filename, 'r') as f:
        yaml_cfg = AttrDict(yaml.load(f))
    _config = _cls2dict(config)
    _merge_a_into_b(yaml_cfg, _config)
    _dict2cls(_config, config)


def merge_cfg_from_list(cfg_list, config):
    """Merge config keys, values in a list (e.g., from command line) into the
    global config. For example, `cfg_list = ['TEST.NMS', 0.5]`.
    """
    _config = _cls2dict(config)
    assert len(cfg_list) % 2 == 0
    for full_key, v in zip(cfg_list[0::2], cfg_list[1::2]):
        # if _key_is_deprecated(full_key):
        #     continue
        # if _key_is_renamed(full_key):
        #     _raise_key_rename_error(full_key)
        key_list = full_key.split('.')
        d = _config
        for subkey in key_list[:-1]:
            assert subkey in d, 'Non-existent key: {}'.format(full_key)
            d = d[subkey]
        subkey = key_list[-1]
        assert subkey in d, 'Non-existent key: {}'.format(full_key)
        value = _decode_cfg_value(v)
        value = _check_and_coerce_cfg_value_type(
            value, d[subkey], subkey, full_key
        )
        d[subkey] = value
    _dict2cls(_config, config)


def _merge_a_into_b(a, b, stack=None):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    assert isinstance(a, AttrDict), 'Argument `a` must be an AttrDict'
    assert isinstance(b, AttrDict), 'Argument `b` must be an AttrDict'

    for k, v_ in a.items():
        full_key = '.'.join(stack) + '.' + k if stack is not None else k
        # a must specify keys that are in b
        if k not in b:
            # if _key_is_deprecated(full_key):
            #     continue
            # elif _key_is_renamed(full_key):
            #     _raise_key_rename_error(full_key)
            # else:
            raise KeyError('Non-existent config key: {}'.format(full_key))

        v = copy.deepcopy(v_)
        v = _decode_cfg_value(v)
        v = _check_and_coerce_cfg_value_type(v, b[k], k, full_key)

        # Recursively merge dicts
        if isinstance(v, AttrDict):
            try:
                stack_push = [k] if stack is None else stack + [k]
                _merge_a_into_b(v, b[k], stack=stack_push)
            except BaseException:
                raise
        else:
            b[k] = v


def _decode_cfg_value(v):
    """Decodes a raw config value (e.g., from a yaml config files or command
    line argument) into a Python object.
    """
    # Configs parsed from raw yaml will contain dictionary keys that need to be
    # converted to AttrDict objects
    if isinstance(v, dict):
        return AttrDict(v)
    # All remaining processing is only applied to strings
    if not isinstance(v, basestring):
        return v
    # Try to interpret `v` as a:
    #   string, number, tuple, list, dict, boolean, or None
    try:
        v = literal_eval(v)
    # The following two excepts allow v to pass through when it represents a
    # string.
    #
    # Longer explanation:
    # The type of v is always a string (before calling literal_eval), but
    # sometimes it *represents* a string and other times a data structure, like
    # a list. In the case that v represents a string, what we got back from the
    # yaml parser is 'foo' *without quotes* (so, not '"foo"'). literal_eval is
    # ok with '"foo"', but will raise a ValueError if given 'foo'. In other
    # cases, like paths (v = 'foo/bar' and not v = '"foo/bar"'), literal_eval
    # will raise a SyntaxError.
    except ValueError:
        pass
    except SyntaxError:
        pass
    return v


def _check_and_coerce_cfg_value_type(value_a, value_b, key, full_key):
    """Checks that `value_a`, which is intended to replace `value_b` is of the
    right type. The type is correct if it matches exactly or is one of a few
    cases in which the type can be easily coerced.
    """
    # The types must match (with some exceptions)
    type_b = type(value_b)
    type_a = type(value_a)
    if type_a is type_b:
        return value_a

    # Exceptions: numpy arrays, strings, tuple<->list
    if isinstance(value_b, np.ndarray):
        value_a = np.array(value_a, dtype=value_b.dtype)
    elif isinstance(value_b, basestring):
        value_a = str(value_a)
    elif isinstance(value_a, tuple) and isinstance(value_b, list):
        value_a = list(value_a)
    elif isinstance(value_a, list) and isinstance(value_b, tuple):
        value_a = tuple(value_a)
    else:
        raise ValueError(
            'Type mismatch ({} vs. {}) with values ({} vs. {}) for config '
            'key: {}'.format(type_b, type_a, value_b, value_a, full_key)
        )
    return value_a


# =========== MODEL UTILITIES ===========
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


def update_config_and_load_model(config, network, train_generator=None):

    choice = config.MODEL.INIT_FILE_CHOICE
    phase = config.CTRL.PHASE

    # determine model_path
    if phase == 'train':
        if os.path.exists(choice):
            print('[{:s}]loading designated weights\t{:s}\n'.format(phase.upper(), choice))
            model_path = choice
            del config.MODEL['PRETRAIN_COCO_MODEL']
            del config.MODEL['PRETRAIN_IMAGENET_MODEL']
        else:
            model_path = _find_last(config)[1]
            if model_path is not None:
                if choice.lower() in ['coco_pretrain', 'imagenet_pretrain']:
                    print('WARNING: find existing model... ignore pretrain model')

                del config.MODEL['PRETRAIN_COCO_MODEL']
                del config.MODEL['PRETRAIN_IMAGENET_MODEL']
            else:
                if choice.lower() == "imagenet_pretrain":
                    model_path = config.MODEL.PRETRAIN_IMAGENET_MODEL
                    suffix = 'imagenet'
                    del config.MODEL['PRETRAIN_COCO_MODEL']
                elif choice.lower() == "coco_pretrain":
                    model_path = config.MODEL.PRETRAIN_COCO_MODEL
                    suffix = 'coco'
                    del config.MODEL['PRETRAIN_IMAGENET_MODEL']
                elif choice.lower() == 'last':
                    model_path = config.MODEL.PRETRAIN_COCO_MODEL
                    suffix = 'coco'
                    del config.MODEL['PRETRAIN_IMAGENET_MODEL']
                    print('init file choice is [LAST]; however no file found; '
                          'use pretrain model to init')
                print('use {:s} pretrain model...'.format(suffix))

        print('loading weights \t{:s}\n'.format(model_path))

    elif phase == 'inference':

        del config.MODEL['PRETRAIN_COCO_MODEL']
        del config.MODEL['PRETRAIN_IMAGENET_MODEL']

        if choice.lower() in ['coco_pretrain', 'imagenet_pretrain', 'last']:
            model_path = _find_last(config)[1]
            print('use last trained model for inference')
        elif os.path.exists(choice):
            model_path = choice
            print('use designated model for inference')
        print('[{:s}] loading model weights\t{:s} for inference\n'.format(phase.upper(), model_path))

    # load model
    checkpoints = torch.load(model_path)
    try:
        network.load_state_dict(checkpoints['state_dict'], strict=False)
    except KeyError:
        network.load_state_dict(checkpoints, strict=False)  # legacy reason

    # determine start_iter and epoch for resume
    # set MODEL.INIT_MODEL
    # update network.start_epoch, network.start_iter
    if phase == 'train':
        try:
            # indicate this is a resumed model
            network.start_epoch = checkpoints['epoch']
            network.start_iter = checkpoints['iter']
            num_train_im = train_generator.dataset.dataset.num_images
            iter_per_epoch = math.floor(num_train_im/config.CTRL.BATCH_SIZE)
            if network.start_iter % iter_per_epoch == 0:
                network.start_iter = 1
                network.start_epoch += 1
            else:
                network.start_iter += 1
        except KeyError:
            # indicate this is a pretrain model
            network.start_epoch, network.start_iter = 1, 1
        # init counters
        network.epoch = network.start_epoch
        network.iter = network.start_iter

    # add new info to config
    config.MODEL.INIT_MODEL = model_path

    # set MISC.LOG_FILE; (inference) MISC.DET_RESULT_FILE, MISC.SAVE_IMAGE_DIR
    if phase == 'train':
        config.MISC.LOG_FILE = os.path.join(config.MISC.RESULT_FOLDER,
                                            'train_log_start_ep_{:04d}_iter_{:06d}.txt'.
                                            format(network.start_epoch, network.start_iter))
        if config.CTRL.DEBUG or config.TRAIN.DO_VALIDATION:
            # set SAVE_IM=True
            config.TEST.SAVE_IM = True
    else:
        model_name = os.path.basename(model_path).replace('.pth', '')   # mask_rcnn_ep_0053_iter_001234
        config.MISC.LOG_FILE = os.path.join(config.MISC.RESULT_FOLDER,
                                            'inference_from_{:s}.txt'.format(model_name))
        model_suffix = os.path.basename(model_path).replace('mask_rcnn_', '')

        config.MISC.DET_RESULT_FILE = os.path.join(config.MISC.RESULT_FOLDER, 'det_result_{:s}'.format(model_suffix))

        if config.TEST.SAVE_IM:
            config.MISC.SAVE_IMAGE_DIR = os.path.join(config.MISC.RESULT_FOLDER, model_suffix.replace('.pth', ''))
            if not os.path.exists(config.MISC.SAVE_IMAGE_DIR):
                os.makedirs(config.MISC.SAVE_IMAGE_DIR)

    config.display(config.MISC.LOG_FILE)
    network.config = config

    return config, network


def set_optimizer(net, opt):

    if opt.OPTIM_METHOD == 'sgd':

        if opt.BN_LEARN:
            parameter_list = [param for name, param in net.named_parameters() if param.requires_grad]
            optimizer = optim.SGD(parameter_list, lr=opt.INIT_LR,   # TODO: direct net.parameters() fail; seek to it.
                                  momentum=opt.MOMENTUM, weight_decay=opt.WEIGHT_DECAY)
        else:
            # Optimizer object, add L2 Regularization
            # Skip gamma and beta weights of batch normalization layers.
            trainables_wo_bn = [param for name, param in net.named_parameters()
                                if param.requires_grad and 'bn' not in name]
            trainables_only_bn = [param for name, param in net.named_parameters()
                                  if param.requires_grad and 'bn' in name]
            optimizer = optim.SGD([
                {'params': trainables_wo_bn, 'weight_decay': opt.WEIGHT_DECAY},
                {'params': trainables_only_bn}
            ], lr=opt.INIT_LR, momentum=opt.MOMENTUM)

    elif opt.OPTIM_METHOD == 'adam':
        optimizer = optim.Adam(net.parameters(), lr=opt.INIT_LR,
                               weight_decay=opt.WEIGHT_DECAY, betas=(0.9, 0.999))
    elif opt.OPTIM_METHOD == 'rmsprop':
        optimizer = optim.RMSprop(net.parameters(), lr=opt.lr,
                                  weight_decay=opt.WEIGHT_DECAY, momentum=opt.MOMENTUM,
                                  alpha=0.9, centered=True)
    return optimizer


def adjust_lr(optimizer, curr_ep, curr_iter, config):

    if config.LR_WARM_UP and curr_ep == 1 and curr_iter <= config.LR_WP_ITER:
        a = config.INIT_LR * (1 - config.LR_WP_FACTOR) / (config.LR_WP_ITER - 1)
        b = config.INIT_LR * config.LR_WP_FACTOR - a
        lr = a * curr_iter + b
    else:
        def _tiny_transfer(schedule):
            out = np.zeros(len(schedule))
            for i in range(len(schedule)):
                out[i] = sum(schedule[:i+1])
            return out
        schedule_list = _tiny_transfer(config.SCHEDULE)
        decay = config.GAMMA ** (sum(curr_ep > schedule_list))
        lr = config.INIT_LR * decay

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

