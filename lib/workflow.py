import os
from tools import visualize
import time
from lib.layers import *
from datasets.pycocotools import mask as maskUtils
from datasets.pycocotools.cocoeval import COCOeval
from tools.utils import print_log, compute_left_time, adjust_lr
import torch.nn as nn
import math
import matplotlib.pyplot as plt
from tools.image_utils import *

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
_TEMP = {'heads': 1, '4+': 2, 'all': 3}


def train_model(input_model, train_generator, valset, optimizer, layers, coco_api=None):
    """
    Args:
        input_model:        nn.DataParallel
        train_generator:    Dataloader
        valset:             Dataset
        lr:                 The learning rate to train with
        layers:
                            (only valid when END2END=False)
                            Allows selecting wich layers to train. It can be:
                                - A regular expression to match layer names to train
                                - One of these predefined values:
                                heads: The RPN, classifier and mask heads of the network
                                all: All the layers
                                3+: Train Resnet stage 3 and up
                                4+: Train Resnet stage 4 and up
                                5+: Train Resnet stage 5 and up
        coco_api            validation api
    """

    stage_name = layers.upper()
    if isinstance(input_model, nn.DataParallel):
        model = input_model.module
    else:
        # single-gpu
        model = input_model

    num_train_im = train_generator.dataset.dataset.num_images
    iter_per_epoch = math.floor(num_train_im/model.config.CTRL.BATCH_SIZE)
    total_ep_till_now = sum(model.config.TRAIN.SCHEDULE[:_TEMP[layers]])

    # check details
    if (num_train_im % model.config.CTRL.BATCH_SIZE) % model.config.MISC.GPU_COUNT != 0:
        print_log('WARNING [TRAIN]: last mini-batch in an epoch is not divisible by gpu number.\n'
                  'total train im: {:d}, batch size: {:d}, gpu num {:d}\n'
                  'last mini-batch size: {:d}\n'.format(
                    num_train_im, model.config.CTRL.BATCH_SIZE, model.config.MISC.GPU_COUNT,
                    (num_train_im % model.config.CTRL.BATCH_SIZE)), model.config.MISC.LOG_FILE)

    if model.epoch > total_ep_till_now:
        print_log('skip {:s} stage ...'.format(stage_name.upper()), model.config.MISC.LOG_FILE)
        return None

    print_log('\n[Current stage: {:s}] start training at epoch {:d}, iter {:d}. \n'
              'Total epoch in this stage: {:d}.'.format(stage_name, model.epoch, model.iter,
                                                        model.config.TRAIN.SCHEDULE[_TEMP[layers]-1]),
              model.config.MISC.LOG_FILE)

    if not model.config.TRAIN.END2END:
        if layers in LAYER_REGEX.keys():
            layers = LAYER_REGEX[layers]
        model.set_trainable(layers)

    # EPOCH LOOP
    for ep in range(model.epoch, total_ep_till_now+1):

        epoch_str = "[Ep {:03d}/{}]".format(ep, total_ep_till_now)
        print_log(epoch_str, model.config.MISC.LOG_FILE)
        # Training
        loss = train_epoch_new(input_model, train_generator, optimizer,
                               stage_name=stage_name, epoch_str=epoch_str,
                               epoch=ep, start_iter=model.iter, total_iter=iter_per_epoch,
                               valset=valset, coco_api=coco_api)
        # Validation (deprecated)
        # val_loss = valid_epoch(val_generator, model.config.VALIDATION_STEPS)

        # TODO (mid): visualize the loss with resume concerned; include visdom
        model.loss_history.append(loss)
        # model.val_loss_history.append(val_loss)
        visualize.plot_loss(model.loss_history, model.val_loss_history,
                            save=True, log_dir=model.config.MISC.RESULT_FOLDER)
        # save model
        model_file = os.path.join(model.config.MISC.RESULT_FOLDER,
                                  'mask_rcnn_ep_{:04d}_iter_{:06d}.pth'.format(ep, iter_per_epoch))
        print_log('Epoch ends, saving model: {:s}\n'.format(model_file), model.config.MISC.LOG_FILE)
        torch.save({
            'state_dict':   model.state_dict(),
            'epoch':        ep,
            'iter':         iter_per_epoch,
        }, model_file)

        # one epoch ends; update iterator
        model.iter = 1
        model.epoch = ep

    # Current stage ends; do validation if possible
    if model.config.TRAIN.DO_VALIDATION:
        print_log('\nDo validation at end of current stage [{:s}] (model ep {:d} iter {:d}) ...'.
                  format(stage_name.upper(), total_ep_till_now, iter_per_epoch), model.config.MISC.LOG_FILE)
        test_model(input_model, valset, coco_api, during_train=True, epoch=ep, iter=iter_per_epoch)
        model.epoch += 1


def train_epoch_new(input_model, data_loader, optimizer, **args):
    """new training flow scheme
    Args:
        input_model
        data_loader
        optimizer
    """
    if isinstance(input_model, nn.DataParallel):
        model = input_model.module
    else:
        # single-gpu
        model = input_model

    config = model.config

    loss_sum = 0
    start_iter, total_iter, curr_ep = args['start_iter'], args['total_iter'], args['epoch']
    actual_total_iter = total_iter - start_iter + 1
    save_iter_base = math.floor(total_iter / config.TRAIN.SAVE_FREQ_WITHIN_EPOCH)

    # create iterator
    data_iterator = iter(data_loader)

    # ITERATION LOOP
    for iter_ind in range(start_iter, total_iter+1):
        
        curr_iter_time_start = time.time()
        lr = adjust_lr(optimizer, curr_ep, iter_ind, config.TRAIN)   # return lr to show in console

        inputs = next(data_iterator)
        images = Variable(inputs[0].cuda())
        image_metas = Variable(inputs[-1].cuda())
        # pad with zeros
        gt_class_ids, gt_boxes, gt_masks, _ = model.adjust_input_gt(inputs[1], inputs[2], inputs[3])

        if config.CTRL.PROFILE_ANALYSIS:
            print('\ncurr_iter: ', iter_ind)
            print('fetch data time: {:.4f}'.format(time.time() - curr_iter_time_start))
            t = time.time()

        # Run object detection
        # [target_rpn_match, rpn_class_logits, target_rpn_bbox, rpn_pred_bbox,
        # target_class_ids, mrcnn_class_logits, target_deltas, mrcnn_bbox, target_mask, mrcnn_mask]
        # the loss shape: gpu_num x 5
        outputs = input_model([images, gt_class_ids, gt_boxes, gt_masks, image_metas], mode='train')
        detailed_loss = torch.mean(outputs, dim=0)
        loss = torch.sum(detailed_loss)

        # Compute losses (moved to forward)
        # loss, detailed_loss = compute_loss(outputs)
        if config.CTRL.PROFILE_ANALYSIS:
            print('forward time: {:.4f}'.format(time.time() - t))
            t = time.time()

        optimizer.zero_grad()
        loss.backward()
        if config.TRAIN.CLIP_GRAD:
            torch.nn.utils.clip_grad_norm(input_model.parameters(), config.TRAIN.MAX_GRAD_NORM)
        optimizer.step()

        if config.CTRL.PROFILE_ANALYSIS:
            print('backward time: {:.4f}'.format(time.time() - t))
            t = time.time()

        # Progress
        if iter_ind % config.CTRL.SHOW_INTERVAL == 0 or iter_ind == args['start_iter']:
            iter_time = time.time() - curr_iter_time_start
            days, hrs = compute_left_time(iter_time, curr_ep,
                                          sum(config.TRAIN.SCHEDULE), iter_ind, total_iter)
            print_log('[{:s}][{:s}]{:s} {:06d}/{} [est. left: {:d} days, {:2.1f} hrs] (iter_t: {:.2f})'
                      '\tlr: {:.6f} | loss: {:.3f} - rpn_cls: {:.3f} - rpn_bbox: {:.3f} '
                      '- mrcnn_cls: {:.3f} - mrcnn_bbox: {:.3f} - mrcnn_mask_loss: {:.3f}'.
                      format(config.CTRL.CONFIG_NAME, args['stage_name'], args['epoch_str'],
                             iter_ind, total_iter,
                             days, hrs, iter_time, lr,
                             loss.data.cpu()[0],
                             detailed_loss[0].data.cpu()[0],
                             detailed_loss[1].data.cpu()[0],
                             detailed_loss[2].data.cpu()[0],
                             detailed_loss[3].data.cpu()[0],
                             detailed_loss[4].data.cpu()[0]),
                      config.MISC.LOG_FILE)
        # Statistics
        loss_sum += loss.data.cpu()[0]/actual_total_iter

        # save model
        if iter_ind % save_iter_base == 0:
            model_file = os.path.join(config.MISC.RESULT_FOLDER,
                                      'mask_rcnn_ep_{:04d}_iter_{:06d}.pth'.format(curr_ep, iter_ind))
            print_log('saving model: {:s}\n'.format(model_file), config.MISC.LOG_FILE)
            torch.save({
                'state_dict':   model.state_dict(),
                'epoch':        curr_ep,        # or model.epoch
                'iter':         iter_ind,       # or model.iter
            }, model_file)

        # for debug; test the model
        # if config.CTRL.DEBUG and iter_ind == (start_iter+100):
        #     print_log('\n[DEBUG] Do validation at stage [{:s}] (model ep {:d} iter {:d}) ...'.
        #               format(args['stage_name'].upper(), args['epoch'], iter_ind), config.MISC.LOG_FILE)
        #     test_model(input_model, args['valset'], args['coco_api'],
        #                during_train=True, epoch=args['epoch'], iter=iter_ind)

    return loss_sum


def test_model(input_model, valset, coco_api,
               limit=-1, image_ids=None, **args):
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

    # set up save and log folder for both train and inference
    if args['during_train']:
        model_file_name = 'mask_rcnn_ep_{:04d}_iter_{:04d}.pth'.format(args['epoch'], args['iter'])
        mode = 'inference'
        _val_folder = model.config.MISC.RESULT_FOLDER.replace('train', 'inference')
        _model_name = model_file_name.replace('.pth', '')
        _model_suffix = _model_name.replace('mask_rcnn_', '')  # say, ep_0053_iter_1234
        log_file = os.path.join(_val_folder, 'inference_from_{:s}.txt'.format(_model_name))
        det_res_file = os.path.join(_val_folder, 'det_result_{:s}.pth'.format(_model_suffix))
        train_log_file = model.config.MISC.LOG_FILE
        save_im_folder = os.path.join(_val_folder, _model_suffix)
        if not os.path.exists(save_im_folder):
            os.makedirs(save_im_folder)
    else:
        # validation-only case
        model_file_name = os.path.basename(model.config.MODEL.INIT_MODEL)
        mode = model.config.CTRL.PHASE
        log_file = model.config.MISC.LOG_FILE
        det_res_file = model.config.MISC.DET_RESULT_FILE
        train_log_file = None
        save_im_folder = model.config.MISC.SAVE_IMAGE_DIR if model.config.TEST.SAVE_IM else None

    dataset = valset.dataset

    # Pick COCO images from the dataset
    image_ids = image_ids or dataset.image_ids
    # Limit to a subset
    if limit > 0:
        image_ids = image_ids[:limit]

    num_test_im = len(image_ids)
    actual_test_bs = model.config.CTRL.BATCH_SIZE * 2  # bs for test could be larger TODO (low): merge info into log

    print("Running COCO evaluation on {} images.".format(num_test_im))
    assert (num_test_im % actual_test_bs) % model.config.MISC.GPU_COUNT == 0, \
        '[INFERENCE] last mini-batch in an epoch is not divisible by gpu number.'
    # Get corresponding COCO image IDs.
    coco_image_ids = [dataset.image_info[ind]["id"] for ind in image_ids]

    t_prediction = 0
    t_start = time.time()

    results, cnt = [], 0
    total_iter = math.ceil(num_test_im / actual_test_bs)
    show_test_progress_base = math.floor(total_iter / (model.config.CTRL.SHOW_INTERVAL/2))
    # note that GPU efficiency is low when SAVE_IM=True

    for iter_ind in range(total_iter):

        curr_start_id = iter_ind*actual_test_bs
        curr_end_id = min(curr_start_id + actual_test_bs, num_test_im)
        curr_image_ids = image_ids[curr_start_id:curr_end_id]

        # Run detection
        t_pred_start = time.time()
        # Mold inputs to format expected by the neural network
        molded_images, image_metas, windows, images = _mold_inputs(model, curr_image_ids, dataset)

        # Run object detection; detections: 8,100,6; mrcnn_mask: 8,100,81,28,28
        detections, mrcnn_mask = input_model([molded_images, image_metas], mode=mode)

        # Convert to numpy
        detections = detections.data.cpu().numpy()
        mrcnn_mask = mrcnn_mask.permute(0, 1, 3, 4, 2).contiguous().data.cpu().numpy()

        # Process detections
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
            if model.config.TEST.SAVE_IM:
                plt.close()
                visualize.display_instances(
                    image, final_rois, final_masks, final_class_ids, CLASS_NAMES, final_scores)

                im_file = os.path.join(save_im_folder, 'coco_im_id_{:d}.png'.format(curr_coco_id))
                plt.savefig(im_file, bbox_inches='tight')

        t_prediction += (time.time() - t_pred_start)
        cnt += len(curr_image_ids)

        # show progress
        if iter_ind % show_test_progress_base == 0 or cnt == len(image_ids):
            print_log('[{:s}][{:s}] evaluation progress \t{:4d} images /{:4d} total ...'.
                      format(model.config.CTRL.CONFIG_NAME, model_file_name, cnt, len(image_ids)),
                      log_file, additional_file=train_log_file)

    print_log("Prediction time: {:.4f}. Average {:.4f} sec/image".format(
        t_prediction, t_prediction / len(image_ids)), log_file, additional_file=train_log_file)
    print_log('Saving results to {:s}'.format(det_res_file), log_file, additional_file=train_log_file)
    torch.save({'det_result': results}, det_res_file)

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
    print_log('Total time: {:.4f}'.format(time.time() - t_start), log_file, additional_file=train_log_file)
    print_log('Config_name [{:s}], model file [{:s}], mAP is {:.4f}\n\n'.
              format(model.config.CTRL.CONFIG_NAME, model_file_name, cocoEval.stats[0]),
              log_file, additional_file=train_log_file)
    print_log('Done!', log_file, additional_file=train_log_file)


# ======================
def compute_loss(inputs):

    target_rpn_match, rpn_class_logits, \
    target_rpn_bbox, rpn_pred_bbox, \
    target_class_ids, mrcnn_class_logits, \
    target_deltas, mrcnn_bbox, \
    target_mask, mrcnn_mask = \
        inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], \
        inputs[5], inputs[6], inputs[7], inputs[8], inputs[9]

    rpn_class_loss = compute_rpn_class_loss(target_rpn_match, rpn_class_logits)
    rpn_bbox_loss = compute_rpn_bbox_loss(target_rpn_bbox, target_rpn_match, rpn_pred_bbox)
    mrcnn_class_loss = compute_mrcnn_class_loss(target_class_ids, mrcnn_class_logits)
    mrcnn_bbox_loss = compute_mrcnn_bbox_loss(target_deltas, target_class_ids, mrcnn_bbox)
    mrcnn_mask_loss = compute_mrcnn_mask_loss(target_mask, target_class_ids, mrcnn_mask)

    outputs = [rpn_class_loss, rpn_bbox_loss, mrcnn_class_loss, mrcnn_bbox_loss, mrcnn_mask_loss]
    return sum(outputs), outputs


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
        molded_image, window, scale, padding = resize_image(
            image, min_dim=model.config.DATA.IMAGE_MIN_DIM,
            max_dim=model.config.DATA.IMAGE_MAX_DIM, padding=model.config.DATA.IMAGE_PADDING)
        molded_image = molded_image.astype(np.float32) - model.config.DATA.MEAN_PIXEL

        # Build image_meta
        image_meta = compose_image_meta(0, image.shape, window,
                                        np.zeros([model.config.DATASET.NUM_CLASSES], dtype=np.int32), 0)
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
    image_metas = Variable(torch.from_numpy(image_metas).cuda(), volatile=True)

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
        full_mask = unmold_mask(masks[i], boxes[i], image_shape)
        full_masks.append(full_mask)
    full_masks = np.stack(full_masks, axis=-1)\
        if full_masks else np.empty((0,) + masks.shape[1:3])

    return boxes, class_ids, scores, full_masks

