import os
import torch.optim as optim
from tools import visualize
import time
from lib.layers import *
from datasets.pycocotools import mask as maskUtils
from datasets.pycocotools.cocoeval import COCOeval
import numpy as np
from tools.utils import print_log
import torch.nn as nn
import matplotlib.pyplot as plt

# Pre-defined layer regular expressions
LAYER_REGEX = {
    # all layers but the backbone
    "heads": r"(fpn.P5\_.*)|(fpn.P4\_.*)|(fpn.P3\_.*)|(fpn.P2\_.*)|(rpn.*)|(classifier.*)|(mask.*)",
    # From a specific Resnet stage and up
    "3+": r"(fpn.C3.*)|(fpn.C4.*)|(fpn.C5.*)|(fpn.P5\_.*)|(fpn.P4\_.*)|"
          r"(fpn.P3\_.*)|(fpn.P2\_.*)|(rpn.*)|(classifier.*)|(mask.*)",
    "4+": r"(fpn.C4.*)|(fpn.C5.*)|(fpn.P5\_.*)|(fpn.P4\_.*)|"
          r"(fpn.P3\_.*)|(fpn.P2\_.*)|(rpn.*)|(classifier.*)|(mask.*)",
    "5+": r"(fpn.C5.*)|(fpn.P5\_.*)|(fpn.P4\_.*)|(fpn.P3\_.*)|(fpn.P2\_.*)|(rpn.*)|(classifier.*)|(mask.*)",
    # All layers
    "all": ".*",
}

CLASS_NAMES = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']


def train_model(input_model, train_generator, val_generator, lr, total_ep_curr_call, layers):
    """
    Args:
        input_model:        nn.DataParallel
        train_generator:
        val_generator:
        lr:
            The learning rate to train with
        total_ep_curr_call:
            Number of training epochs. Note that previous training epochs
            are considered to be done alreay, so this actually determines
            the epochs to train in total rather than in this particaular call.
        layers:
            Allows selecting wich layers to train. It can be:
                - A regular expression to match layer names to train
                - One of these predefined values:
                heads: The RPN, classifier and mask heads of the network
                all: All the layers
                3+: Train Resnet stage 3 and up
                4+: Train Resnet stage 4 and up
                5+: Train Resnet stage 5 and up
    """
    stage_name = layers.upper()
    if isinstance(input_model, nn.DataParallel):
        model = input_model.module
    else:
        # single-gpu
        model = input_model

    num_train_im = train_generator.dataset.dataset.num_images
    if (num_train_im % model.config.BATCH_SIZE) % model.config.GPU_COUNT != 0:
        print_log('WARNING: last mini-batch in an epoch is not divisible by gpu number.\n'
                  'total train im: {:d}, batch size: {:d}, gpu num {:d}\n'
                  'last mini-batch size: {:d}\n'.format(
            num_train_im, model.config.BATCH_SIZE, model.config.GPU_COUNT,
            (num_train_im % model.config.BATCH_SIZE)),
            model.config.LOG_FILE)

    if model.epoch > total_ep_curr_call:
        print_log('skip {:s} stage ...'.format(stage_name), model.config.LOG_FILE)
        return None

    if layers in LAYER_REGEX.keys():
        layers = LAYER_REGEX[layers]
    model.set_trainable(layers)

    # Optimizer object, add L2 Regularization
    # Skip gamma and beta weights of batch normalization layers.
    trainables_wo_bn = [param for name, param in model.named_parameters() if param.requires_grad and 'bn' not in name]
    trainables_only_bn = [param for name, param in model.named_parameters() if param.requires_grad and 'bn' in name]
    optimizer = optim.SGD([
        {'params': trainables_wo_bn, 'weight_decay': model.config.WEIGHT_DECAY},
        {'params': trainables_only_bn}
    ], lr=lr, momentum=model.config.LEARNING_MOMENTUM)

    print_log('\n[Current stage: {:s}] start training at epoch {:d}, iter {:d}. \n'
              'Total epoch this stage: {:d}, LR={:.4f}'.format(
                stage_name, model.epoch+1, model.iter+1, total_ep_curr_call, lr),
                model.config.LOG_FILE)

    for epoch in range(model.epoch+1, total_ep_curr_call+1):

        epoch_str = "[Epoch {}/{}]".format(epoch, total_ep_curr_call)
        print_log(epoch_str, model.config.LOG_FILE)
        # Training
        if model.config.old_scheme:
            loss = train_epoch(input_model, train_generator, optimizer,
                               model.config.STEPS_PER_EPOCH, stage_name, epoch_str)
        else:
            loss = train_epoch_new(input_model, train_generator, optimizer,
                                   stage_name=stage_name, epoch_str=epoch_str,
                                   epoch=epoch, start_iter=model.iter+1)
        # Validation
        # val_loss = valid_epoch(val_generator, model.config.VALIDATION_STEPS)

        # Statistics
        model.loss_history.append(loss)
        # model.val_loss_history.append(val_loss)
        visualize.plot_loss(model.loss_history, model.val_loss_history, save=True, log_dir=model.log_dir)
        model_file = model.checkpoint_path.format(epoch)
        print_log('saving model: {:s}\n'.format(model_file), model.config.LOG_FILE)
        torch.save({'state_dict': model.state_dict()}, model_file)
        model.iter = 0

    # update the epoch info
    model.epoch = total_ep_curr_call


def train_epoch_new(input_model, data_loader, optimizer, **args):
    """new training flow scheme"""
    if isinstance(input_model, nn.DataParallel):
        model = input_model.module
    else:
        # single-gpu
        model = input_model

    loss_sum = 0
    config = model.config
    data_iterator = iter(data_loader)
    iter_per_epoch = math.ceil(len(data_loader)/config.BATCH_SIZE)
    save_iter_base = math.floor(iter_per_epoch / config.SAVE_TIME_WITHIN_EPOCH)

    for iter_ind in range(args['start_iter'], iter_per_epoch+1):

        inputs = next(data_iterator)

        images = Variable(inputs[0].cuda())
        target_rpn_match = Variable(inputs[2].cuda())
        target_rpn_bbox = Variable(inputs[3].cuda())
        # pad with zeros
        gt_class_ids, gt_boxes, gt_masks, _ = model.adjust_input_gt(inputs[4], inputs[5], inputs[6])

        # Run object detection
        # [rpn_class_logits, rpn_pred_bbox,
        # target_class_ids, mrcnn_class_logits, target_deltas, mrcnn_bbox, target_mask, mrcnn_mask]
        outputs = input_model([images, gt_class_ids, gt_boxes, gt_masks], mode=model.config.PHASE)

        # Compute losses
        loss, detailed_losses = compute_loss(target_rpn_match, target_rpn_bbox, outputs)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm(input_model.parameters(), 5.0)
        optimizer.step()

        # Progress
        if iter_ind % model.config.SHOW_INTERVAL == 0 or iter_ind == args['start_iter']:
            print_log('[{:s}][stage {:s}]{:s}\t{}/{}\tloss: {:.5f} - rpn_cls: {:.5f} - rpn_bbox: {:.5f} '
                      '- mrcnn_cls: {:.5f} - mrcnn_bbox: {:.5f} - mrcnn_mask_loss: {:.5f}'.
                      format(model.config.NAME, args['stage_name'], args['epoch_str'], iter_ind+1, iter_per_epoch,
                             loss.data.cpu()[0],
                             detailed_losses[0].data.cpu()[0],
                             detailed_losses[1].data.cpu()[0],
                             detailed_losses[2].data.cpu()[0],
                             detailed_losses[3].data.cpu()[0],
                             detailed_losses[4].data.cpu()[0]), config.LOG_FILE)
        # Statistics
        loss_sum += loss.data.cpu()[0]/iter_per_epoch
        if iter_ind % save_iter_base == 0:
            model_file = os.path.join(model.log_dir,
                                      'mask_rcnn_{:04d}_iter_{:d}.pth'.format(args['epoch'], iter_ind))
            print_log('saving model file to: {:s}'.format(model_file), config.LOG_FILE)
            torch.save({
                'state_dict':   model.state_dict(),
                'epoch':        model.epoch,
                'iter':         iter_ind,
            }, model_file)

    return loss_sum


def test_model(input_model, valset, coco_api, limit=-1, image_ids=None):
    """
        Test the trained model
        Args:
            input_model:    nn.DataParallel
            valset:         validation dataset
            coco_api:       api
            limit:          the number of images to use for evaluation
            image_ids:      a certain image
    """
    if isinstance(input_model, nn.DataParallel):
        model = input_model.module
    else:
        # single-gpu
        model = input_model

    model_file_name = os.path.basename(model.config.START_MODEL_FILE)
    dataset = valset.dataset

    # Pick COCO images from the dataset
    image_ids = image_ids or dataset.image_ids
    # Limit to a subset
    if limit > 0:
        image_ids = image_ids[:limit]

    num_test_im = len(image_ids)
    print("Running COCO evaluation on {} images.".format(num_test_im))
    assert (num_test_im % model.config.BATCH_SIZE) % model.config.GPU_COUNT == 0, 'last mini-batch in an epoch' \
                                                                                  'is not divisible by gpu number.'
    # Get corresponding COCO image IDs.
    coco_image_ids = [dataset.image_info[ind]["id"] for ind in image_ids]

    t_prediction = 0
    t_start = time.time()

    results = []
    total_iter = math.ceil(num_test_im / model.config.BATCH_SIZE)
    cnt = 0

    # for i, image_id in enumerate(image_ids):
    for iter_ind in range(total_iter):
        curr_image_ids = image_ids[iter_ind*model.config.BATCH_SIZE :
                            min(iter_ind*model.config.BATCH_SIZE + model.config.BATCH_SIZE, num_test_im)]

        # Run detection
        t_pred_start = time.time()
        # Mold inputs to format expected by the neural network
        molded_images, image_metas, windows, images = _mold_inputs(model, curr_image_ids, dataset)

        # Run object detection; detections: 8,100,6; mrcnn_mask: 8,100,81,28,28
        detections, mrcnn_mask = input_model([molded_images, image_metas], mode=model.config.PHASE)

        # Convert to numpy
        detections = detections.data.cpu().numpy()
        mrcnn_mask = mrcnn_mask.permute(0, 1, 3, 4, 2).data.cpu().numpy()

        # Process detections
        results = []
        for i, image in enumerate(images):

            curr_coco_id = coco_image_ids[curr_image_ids[i]]
            final_rois, final_class_ids, final_scores, final_masks = _unmold_detections(
                detections[i], mrcnn_mask[i], image.shape, windows[i])

            if final_rois is None:
                continue
            for det_id in range(final_rois.shape[0]):
                bbox = np.around(final_rois[det_id], 1)
                curr_result = {
                    "image_id":     curr_coco_id,
                    "category_id":  dataset.get_source_class_id(final_class_ids[det_id], "coco"),
                    "bbox":         [bbox[1], bbox[0], bbox[3] - bbox[1], bbox[2] - bbox[0]],
                    "score":        final_scores[det_id],
                    "segmentation": maskUtils.encode(np.asfortranarray(final_masks[:, :, det_id]))
                }
                results.append(curr_result)

            # visualize result if necessary
            if model.config.DEBUG:
                plt.close()
                visualize.display_instances(image, final_rois, final_masks, final_class_ids,
                                            CLASS_NAMES, final_scores)
                im_file = os.path.join(model.config.SAVE_IMAGE_DIR,
                                       'coco_im_id_{:d}.png'.format(curr_coco_id))
                plt.savefig(im_file)

        t_prediction += (time.time() - t_pred_start)
        cnt += len(curr_image_ids)
        if iter_ind % (model.config.SHOW_INTERVAL*10) == 0 or cnt == len(image_ids):
            print_log('[{:s}][{:s}] evaluation progress \t{:4d} images /{:4d} total ...'.
                      format(model.config.NAME, model_file_name, cnt, len(image_ids)), model.config.LOG_FILE)

    print_log("Prediction time: {:.4f}. Average {:.4f} sec/image".format(
        t_prediction, t_prediction / len(image_ids)), model.config.LOG_FILE)
    print_log('Saving results to {:s}'.format(model.config.RESULT_FILE), model.config.LOG_FILE)
    torch.save({'det_result': results}, model.config.RESULT_FILE)

    # Evaluate
    print('\nBegin to evaluate ...')
    # Load results. This modifies results with additional attributes.
    coco_results = coco_api.loadRes(results)
    eval_type = "bbox"
    cocoEval = COCOeval(coco_api, coco_results, eval_type)
    cocoEval.params.imgIds = coco_image_ids
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    print_log('Total time: {:.4f}'.format(time.time() - t_start), model.config.LOG_FILE)
    print_log('config [{:s}], model file [{:s}], mAP is {:.4f}\n\n'.
              format(model.config.NAME, os.path.basename(model.config.START_MODEL_FILE, cocoEval.stats[0])),
              model.config.LOG_FILE)


def compute_loss(target_rpn_match, target_rpn_bbox, inputs):

    rpn_class_logits, rpn_pred_bbox, target_class_ids, \
        mrcnn_class_logits, target_deltas, mrcnn_bbox, target_mask, mrcnn_mask = \
        inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5], inputs[6], inputs[7]

    rpn_class_loss = compute_rpn_class_loss(target_rpn_match, rpn_class_logits)
    rpn_bbox_loss = compute_rpn_bbox_loss(target_rpn_bbox, target_rpn_match, rpn_pred_bbox)
    mrcnn_class_loss = compute_mrcnn_class_loss(target_class_ids, mrcnn_class_logits)
    mrcnn_bbox_loss = compute_mrcnn_bbox_loss(target_deltas, target_class_ids, mrcnn_bbox)
    mrcnn_mask_loss = compute_mrcnn_mask_loss(target_mask, target_class_ids, mrcnn_mask)

    outputs = [rpn_class_loss, rpn_bbox_loss, mrcnn_class_loss, mrcnn_bbox_loss, mrcnn_mask_loss]
    return sum(outputs), outputs


def train_epoch(model, datagenerator, optimizer, steps, stage_name, epoch_str):
    batch_count, loss_sum, step = 0, 0, 0

    for inputs in datagenerator:

        batch_count += 1

        images = Variable(inputs[0]).cuda()
        image_metas = inputs[1].numpy()
        gt_class_ids = inputs[4]
        gt_boxes = inputs[5]
        gt_masks = inputs[6]

        # Run object detection
        outputs = \
            model([images, image_metas, gt_class_ids, gt_boxes, gt_masks], mode='training')

        # Compute losses
        rpn_match = Variable(inputs[2]).cuda()
        rpn_bbox = Variable(inputs[3]).cuda()
        loss, detailed_losses = compute_loss(rpn_match, rpn_bbox, outputs)

        # backprop
        if (batch_count % model.config.BATCH_SIZE) == 0:
            optimizer.zero_grad()
        # TODO: no average here?
        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), 5.0)
        if (batch_count % model.config.BATCH_SIZE) == 0:
            optimizer.step()
            batch_count = 0

        # Progress
        if step % model.config.SHOW_INTERVAL == 0:
            print_log('[{:s}][stage {:s}]{:s}\t{}/{}\tloss: {:.5f} - rpn_cls: {:.5f} - rpn_bbox: {:.5f} '
                      '- mrcnn_cls: {:.5f} - mrcnn_bbox: {:.5f} - mrcnn_mask_loss: {:.5f}'.
                      format(model.config.NAME, stage_name, epoch_str, step+1, steps,
                             loss.data.cpu()[0],
                             detailed_losses[0].data.cpu()[0],
                             detailed_losses[1].data.cpu()[0],
                             detailed_losses[2].data.cpu()[0],
                             detailed_losses[3].data.cpu()[0],
                             detailed_losses[4].data.cpu()[0]), model.config.LOG_FILE)
        # Statistics
        loss_sum += loss.data.cpu()[0]/steps

        # Break after 'steps' steps
        # TODO: default steps - 16000; hence each epoch has the same first 16000 images to train?
        if step == steps-1:
            break
        step += 1
    return loss_sum


def _find_last(config, model_dir):
    # Get directory names. Each directory corresponds to a model
    dir_names = next(os.walk(model_dir))[1]
    key = config.NAME.lower()
    dir_names = filter(lambda f: f.startswith(key), dir_names)
    dir_names = sorted(dir_names)
    if not dir_names:
        return None, None
    # Pick last directory
    dir_name = os.path.join(model_dir, dir_names[-1], 'train')
    # Find the last checkpoint
    checkpoints = next(os.walk(dir_name))[2]
    checkpoints = filter(lambda f: f.startswith("mask_rcnn"), checkpoints)
    checkpoints = sorted(checkpoints)
    if not checkpoints:
        return dir_name, None
    checkpoint = os.path.join(dir_name, checkpoints[-1])
    return dir_name, checkpoint


def select_weights(config, network, model_dir):

    choice = config.MODEL_FILE_CHOICE
    phase = config.PHASE

    if phase == 'train':

        if os.path.exists(choice):
            print('[{:s}]loading designated weights\t{:s}\n'.format(phase.upper(), choice))
            model_path = choice
        else:
            model_path = _find_last(config, model_dir)[1]
            if model_path is not None:
                if choice.lower() in ['coco_pretrain', 'imagenet_pretrain']:
                    print('WARNING: find existing model... ignore pretrain model')
            else:
                if choice.lower() == "imagenet_pretrain":
                    model_path = config.PRETRAIN_IMAGENET_MODEL_PATH
                    suffix = 'imagenet'
                elif choice.lower() == "coco_pretrain":
                    model_path = config.PRETRAIN_COCO_MODEL_PATH
                    suffix = 'coco'
                print('use {:s} pretrain model...'.format(suffix))

        print('loading weights \t{:s}\n'.format(model_path))

    elif phase == 'inference':
        if choice.lower() in ['coco_pretrain', 'imagenet_pretrain', 'last']:
            model_path = _find_last(config, model_dir)[1]
            print('use last trained model for inference')
        elif os.path.exists(choice):
            model_path = choice
            print('use designated model for inference')
        print('[{:s}] loading model weights\t{:s} for inference\n'.format(phase.upper(), model_path))

    network.load_weights(model_path)
    # add new info to config
    config.START_MODEL_FILE = model_path

    if config.PHASE == 'train':
        config.START_EPOCH = network.start_epoch
        config.START_ITER = network.start_iter
        config.LOG_FILE = os.path.join(
            network.log_dir, 'log_start_ep_{:d}_iter_{:d}.txt'.format(network.start_epoch, network.start_iter))
    else:
        model_name = os.path.basename(model_path).replace('.pth', '')
        config.LOG_FILE = os.path.join(
            network.log_dir, 'inference_{:s}.txt'.format(model_name))
        model_suffix = os.path.basename(config.START_MODEL_FILE).replace('mask_rcnn_', '')
        config.RESULT_FILE = os.path.join(network.log_dir, 'detection_result_{:s}'.format(model_suffix))
        config.SAVE_IMAGE_DIR = os.path.join(network.log_dir, model_suffix.replace('.pth', ''))
        if not os.path.exists(config.SAVE_IMAGE_DIR):
            os.makedirs(config.SAVE_IMAGE_DIR)

    config.CHECKPOINT_PATH = network.checkpoint_path
    return config


# TODO(low: valid epoch during training)
# def valid_epoch(model, datagenerator, steps):
#
#     step, loss_sum = 0, 0
#
#     for inputs in datagenerator:
#         images = inputs[0]
#         image_metas = inputs[1]
#         rpn_match = inputs[2]
#         rpn_bbox = inputs[3]
#         gt_class_ids = inputs[4]
#         gt_boxes = inputs[5]
#         gt_masks = inputs[6]
#
#         # image_metas as numpy array
#         image_metas = image_metas.numpy()
#
#         # Wrap in variables
#         images = Variable(images, volatile=True)
#         rpn_match = Variable(rpn_match, volatile=True)
#         rpn_bbox = Variable(rpn_bbox, volatile=True)
#         gt_class_ids = Variable(gt_class_ids, volatile=True)
#         gt_boxes = Variable(gt_boxes, volatile=True)
#         gt_masks = Variable(gt_masks, volatile=True)
#
#         # To GPU
#         if self.config.GPU_COUNT:
#             images = images.cuda()
#             rpn_match = rpn_match.cuda()
#             rpn_bbox = rpn_bbox.cuda()
#             gt_class_ids = gt_class_ids.cuda()
#             gt_boxes = gt_boxes.cuda()
#             gt_masks = gt_masks.cuda()
#
#         # Run object detection
#         rpn_class_logits, rpn_pred_bbox, target_class_ids, mrcnn_class_logits, \
#             target_deltas, mrcnn_bbox, target_mask, mrcnn_mask = \
#             self.predict([images, image_metas, gt_class_ids, gt_boxes, gt_masks], mode='training')
#
#         if not target_class_ids.size():
#             continue
#
#         # Compute losses
#         rpn_class_loss, rpn_bbox_loss, mrcnn_class_loss, mrcnn_bbox_loss, mrcnn_mask_loss = \
#             compute_losses(rpn_match, rpn_bbox, rpn_class_logits, rpn_pred_bbox, target_class_ids,
#                            mrcnn_class_logits, target_deltas, mrcnn_bbox, target_mask, mrcnn_mask)
#         loss = rpn_class_loss + rpn_bbox_loss + mrcnn_class_loss + mrcnn_bbox_loss + mrcnn_mask_loss
#
#         # Progress
#         utils.printProgressBar(step + 1, steps, prefix="\t{}/{}".format(step + 1, steps),
#                                suffix="Complete - loss: {:.5f} - rpn_class_loss: {:.5f} - rpn_bbox_loss: {:.5f} - "
#                                       "mrcnn_class_loss: {:.5f} - mrcnn_bbox_loss: {:.5f} - "
#                                       "mrcnn_mask_loss: {:.5f}".format(
#                                    loss.data.cpu()[0],
#                                    rpn_class_loss.data.cpu()[0], rpn_bbox_loss.data.cpu()[0],
#                                    mrcnn_class_loss.data.cpu()[0], mrcnn_bbox_loss.data.cpu()[0],
#                                    mrcnn_mask_loss.data.cpu()[0]), length=10)
#         # Statistics
#         loss_sum += loss.data.cpu()[0]/steps
#
#         # Break after 'steps' steps
#         if step == steps-1:
#             break
#         step += 1
#
#     return loss_sum


def _mold_inputs(model, image_ids, dataset):
    """
        FOR EVALUATION ONLY.
        Takes a list of images and modifies them to the format expected as an input to the neural network.
        images: List of image matrices [height,width,depth]. Images can have different sizes.

        Returns 3 Numpy matrices:
            molded_images: [N, h, w, 3]. Images resized and normalized.
            image_metas: [N, length of meta datasets]. Details about each image.
            windows: [N, (y1, x1, y2, x2)]. The portion of the image that has the
            original image (padding excluded).
    """
    molded_images = []
    image_metas = []
    windows = []
    images = []

    for curr_id in image_ids:
        image = dataset.load_image(curr_id)
        # Resize image to fit the model expected size
        molded_image, window, scale, padding = utils.resize_image(
            image,
            min_dim=model.config.IMAGE_MIN_DIM,
            max_dim=model.config.IMAGE_MAX_DIM,
            padding=model.config.IMAGE_PADDING)
        molded_image = utils.mold_image(molded_image, model.config)
        # Build image_meta
        image_meta = utils.compose_image_meta(
            0, image.shape, window,
            np.zeros([model.config.NUM_CLASSES], dtype=np.int32))
        # Append
        molded_images.append(molded_image)
        windows.append(window)
        image_metas.append(image_meta)
        images.append(image)
    # Pack into arrays
    molded_images = np.stack(molded_images)
    image_metas = np.stack(image_metas)
    windows = np.stack(windows)

    # Convert images to torch tensor
    molded_images = torch.from_numpy(molded_images.transpose(0, 3, 1, 2)).float()
    molded_images = Variable(molded_images.cuda(), volatile=True)

    return molded_images, image_metas, windows, images


def _unmold_detections(detections, mrcnn_mask, image_shape, window):
    """
        FOR EVALUATION ONLY.
        Re-formats the detections of one image from the format of the neural
        network output to a format suitable for use in the rest of the application.

            detections:     [N, (y1, x1, y2, x2, class_id, score)]
            mrcnn_mask:     [N, height, width, num_classes]
            image_shape:    [height, width, depth] Original size of the image before resizing
            window:         [y1, x1, y2, x2] Box in the image where the real image is excluding the padding.

        Returns:
            boxes:          [N, (y1, x1, y2, x2)] Bounding boxes in pixels
            class_ids:      [N] Integer class IDs for each bounding box
            scores:         [N] Float probability scores of the class_id
            masks:          [height, width, num_instances] Instance masks
    """
    # TODO: (low) consider the batch size dim
    # How many detections do we have?
    # Detections array is padded with zeros. Find the first class_id == 0.
    zero_ix = np.where(detections[:, 4] == 0)[0]
    N = zero_ix[0] if zero_ix.shape[0] > 0 else detections.shape[0]

    # Extract boxes, class_ids, scores, and class-specific masks
    boxes = detections[:N, :4]
    class_ids = detections[:N, 4].astype(np.int32)
    scores = detections[:N, 5]
    masks = mrcnn_mask[np.arange(N), :, :, class_ids]

    # Compute scale and shift to translate coordinates to image domain.
    h_scale = image_shape[0] / (window[2] - window[0])
    w_scale = image_shape[1] / (window[3] - window[1])
    scale = min(h_scale, w_scale)
    shift = window[:2]  # y, x
    scales = np.array([scale, scale, scale, scale])
    shifts = np.array([shift[0], shift[1], shift[0], shift[1]])

    # Translate bounding boxes to image domain
    boxes = np.multiply(boxes - shifts, scales).astype(np.int32)

    # Filter out detections with zero area. Often only happens in early
    # stages of training when the network weights are still a bit random.
    exclude_ix = np.where(
        (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) <= 0)[0]
    if exclude_ix.shape[0] > 0:
        boxes = np.delete(boxes, exclude_ix, axis=0)
        class_ids = np.delete(class_ids, exclude_ix, axis=0)
        scores = np.delete(scores, exclude_ix, axis=0)
        masks = np.delete(masks, exclude_ix, axis=0)
        N = class_ids.shape[0]

    # Resize masks to original image size and set boundary threshold.
    full_masks = []
    for i in range(N):
        # Convert neural network mask to full size mask
        full_mask = utils.unmold_mask(masks[i], boxes[i], image_shape)
        full_masks.append(full_mask)
    full_masks = np.stack(full_masks, axis=-1)\
        if full_masks else np.empty((0,) + masks.shape[1:3])

    return boxes, class_ids, scores, full_masks

