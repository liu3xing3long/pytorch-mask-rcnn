import os
import torch
import torch.nn.functional as F
import torch.utils.data
from torch.autograd import Variable
import tools.utils as utils
import torch.optim as optim
from tools import visualize
import time
from datasets.pycocotools import mask as maskUtils
from datasets.pycocotools.cocoeval import COCOeval
import numpy as np

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


def train_model(model, train_generator, val_generator,
                lr, total_ep_curr_call, layers):
    """
    train_dataset, val_dataset:
            Training and validation Dataset objects.
    learning_rate:
            The learning rate to train with
    epochs:
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
    if model.epoch > total_ep_curr_call:
        print('skip {:s} stage ...'.format(stage_name))
        return None

    if layers in LAYER_REGEX.keys():
        layers = LAYER_REGEX[layers]

    model.set_trainable(layers)
    # original data generator here. [MOVED to main.py]
    # Train
    utils.log('\nStarting at epoch {}. LR={}'.format(model.epoch+1, lr))
    utils.log('Checkpoint Path: {}'.format(model.checkpoint_path))

    # Optimizer object
    # Add L2 Regularization
    # Skip gamma and beta weights of batch normalization layers.
    trainables_wo_bn = [param for name, param in model.named_parameters() if param.requires_grad and not 'bn' in name]
    trainables_only_bn = [param for name, param in model.named_parameters() if param.requires_grad and 'bn' in name]
    optimizer = optim.SGD([
        {'params': trainables_wo_bn, 'weight_decay': model.config.WEIGHT_DECAY},
        {'params': trainables_only_bn}
    ], lr=lr, momentum=model.config.LEARNING_MOMENTUM)

    for epoch in range(model.epoch+1, total_ep_curr_call+1):

        utils.log("Epoch {}/{}.".format(epoch, total_ep_curr_call))
        # Training
        loss = train_epoch(model, train_generator, optimizer,
                           model.config.STEPS_PER_EPOCH, stage_name)
        # Validation
        # val_loss = valid_epoch(val_generator, model.config.VALIDATION_STEPS)
        # Statistics
        model.loss_history.append(loss)
        # model.val_loss_history.append(val_loss)
        visualize.plot_loss(model.loss_history, model.val_loss_history, save=True, log_dir=model.log_dir)
        # Save model
        torch.save(model.state_dict(), model.checkpoint_path.format(epoch))

    # update the epoch info
    model.epoch = total_ep_curr_call


def train_epoch(model, datagenerator, optimizer, steps, stage_name):
    batch_count, loss_sum, step = 0, 0, 0
    # while True:
    for inputs in datagenerator:
        # inputs = next(datagenerator)
        batch_count += 1

        images = Variable(inputs[0]).cuda()
        image_metas = inputs[1].numpy()
        gt_class_ids = Variable(inputs[4]).cuda()
        gt_boxes = Variable(inputs[5]).cuda()
        gt_masks = Variable(inputs[6]).cuda()

        # Run object detection
        outputs = \
            model([images, image_metas, gt_class_ids, gt_boxes, gt_masks], mode='training')

        # Compute losses
        rpn_match = Variable(inputs[2]).cuda()
        rpn_bbox = Variable(inputs[3]).cuda()
        loss, detailed_losses = compute_losses(rpn_match, rpn_bbox, outputs)

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
        # if step % 1 == 0:
        if step % model.config.SHOW_INTERVAL == 0:
            utils.printProgressBar(step+1, steps,
                                   prefix="\t[stage {:s}]\t{}/{}".format(stage_name, step+1, steps),
                                   suffix="Complete - loss: {:.5f} - rpn_class_loss: {:.5f} - rpn_bbox_loss: {:.5f} "
                                          "- mrcnn_class_loss: {:.5f} - mrcnn_bbox_loss: {:.5f} "
                                          "- mrcnn_mask_loss: {:.5f}".format(
                                       loss.data.cpu()[0],
                                       detailed_losses[0].data.cpu()[0],
                                       detailed_losses[1].data.cpu()[0],
                                       detailed_losses[2].data.cpu()[0],
                                       detailed_losses[3].data.cpu()[0],
                                       detailed_losses[4].data.cpu()[0]),
                                   length=10)
        # Statistics
        loss_sum += loss.data.cpu()[0]/steps

        # Break after 'steps' steps
        # TODO: default steps - 16000; hence each epoch has the same first 16000 images to train?
        if step == steps-1:
            break
        step += 1

    return loss_sum


def find_last(config, model_dir):
    """
        Finds the last checkpoint file of the last trained model in the model directory.
        Returns:
            log_dir:            The directory where events and weights are saved
            checkpoint_path:    The path to the last checkpoint file
    """
    # Get directory names. Each directory corresponds to a model
    dir_names = next(os.walk(model_dir))[1]
    key = config.NAME.lower() + '_2018'
    dir_names = filter(lambda f: f.startswith(key), dir_names)
    dir_names = sorted(dir_names)
    if not dir_names:
        return None, None
    # Pick last directory
    dir_name = os.path.join(model_dir, dir_names[-1])
    # Find the last checkpoint
    checkpoints = next(os.walk(dir_name))[2]
    checkpoints = filter(lambda f: f.startswith("mask_rcnn"), checkpoints)
    checkpoints = sorted(checkpoints)
    if not checkpoints:
        return dir_name, None
    checkpoint = os.path.join(dir_name, checkpoints[-1])
    return dir_name, checkpoint


############################################################
#  Loss Functions
############################################################
def compute_rpn_class_loss(rpn_match, rpn_class_logits):
    """RPN anchor classifier loss.

    rpn_match: [batch, anchors, 1]. Anchor match type. 1=positive,
               -1=negative, 0=neutral anchor.
    rpn_class_logits: [batch, anchors, 2]. RPN classifier logits for FG/BG.
    """

    # Squeeze last dim to simplify
    rpn_match = rpn_match.squeeze(2)

    # Get anchor classes. Convert the -1/+1 match to 0/1 values.
    anchor_class = (rpn_match == 1).long()

    # Positive and Negative anchors contribute to the loss,
    # but neutral anchors (match value = 0) don't.
    indices = torch.nonzero(rpn_match != 0)

    # Pick rows that contribute to the loss and filter out the rest.
    rpn_class_logits = rpn_class_logits[indices.data[:,0],indices.data[:,1],:]
    anchor_class = anchor_class[indices.data[:,0],indices.data[:,1]]

    # Crossentropy loss
    loss = F.cross_entropy(rpn_class_logits, anchor_class)

    return loss


def compute_rpn_bbox_loss(target_bbox, rpn_match, rpn_bbox):
    """Return the RPN bounding box loss graph.

    target_bbox: [batch, max positive anchors, (dy, dx, log(dh), log(dw))].
        Uses 0 padding to fill in unsed bbox deltas.
    rpn_match: [batch, anchors, 1]. Anchor match type. 1=positive,
               -1=negative, 0=neutral anchor.
    rpn_bbox: [batch, anchors, (dy, dx, log(dh), log(dw))]
    """

    # Squeeze last dim to simplify
    rpn_match = rpn_match.squeeze(2)

    # Positive anchors contribute to the loss, but negative and
    # neutral anchors (match value of 0 or -1) don't.
    indices = torch.nonzero(rpn_match==1)

    # Pick bbox deltas that contribute to the loss
    rpn_bbox = rpn_bbox[indices.data[:,0],indices.data[:,1]]

    # Trim target bounding box deltas to the same length as rpn_bbox.
    target_bbox = target_bbox[0,:rpn_bbox.size()[0],:]

    # Smooth L1 loss
    loss = F.smooth_l1_loss(rpn_bbox, target_bbox)

    return loss


def compute_mrcnn_class_loss(target_class_ids, pred_class_logits):
    """Loss for the classifier head of Mask RCNN.

    target_class_ids: [batch, num_rois]. Integer class IDs. Uses zero
        padding to fill in the array.
    pred_class_logits: [batch, num_rois, num_classes]
    """

    # Loss
    if target_class_ids.size():
        loss = F.cross_entropy(pred_class_logits,target_class_ids.long())
    else:
        loss = Variable(torch.FloatTensor([0]), requires_grad=False)
        if target_class_ids.is_cuda:
            loss = loss.cuda()

    return loss


def compute_mrcnn_bbox_loss(target_bbox, target_class_ids, pred_bbox):
    """Loss for Mask R-CNN bounding box refinement.

    target_bbox: [batch, num_rois, (dy, dx, log(dh), log(dw))]
    target_class_ids: [batch, num_rois]. Integer class IDs.
    pred_bbox: [batch, num_rois, num_classes, (dy, dx, log(dh), log(dw))]
    """

    if target_class_ids.size():
        # Only positive ROIs contribute to the loss. And only
        # the right class_id of each ROI. Get their indicies.
        positive_roi_ix = torch.nonzero(target_class_ids > 0)[:, 0]
        positive_roi_class_ids = target_class_ids[positive_roi_ix.data].long()
        indices = torch.stack((positive_roi_ix,positive_roi_class_ids), dim=1)

        # Gather the deltas (predicted and true) that contribute to loss
        target_bbox = target_bbox[indices[:,0].data,:]
        pred_bbox = pred_bbox[indices[:,0].data,indices[:,1].data,:]

        # Smooth L1 loss
        loss = F.smooth_l1_loss(pred_bbox, target_bbox)
    else:
        loss = Variable(torch.FloatTensor([0]), requires_grad=False)
        if target_class_ids.is_cuda:
            loss = loss.cuda()

    return loss


def compute_mrcnn_mask_loss(target_masks, target_class_ids, pred_masks):
    """Mask binary cross-entropy loss for the masks head.

    target_masks: [batch, num_rois, height, width].
        A float32 tensor of values 0 or 1. Uses zero padding to fill array.
    target_class_ids: [batch, num_rois]. Integer class IDs. Zero padded.
    pred_masks: [batch, proposals, height, width, num_classes] float32 tensor
                with values from 0 to 1.
    """
    if target_class_ids.size():
        # Only positive ROIs contribute to the loss. And only
        # the class specific mask of each ROI.
        positive_ix = torch.nonzero(target_class_ids > 0)[:, 0]
        positive_class_ids = target_class_ids[positive_ix.data].long()
        indices = torch.stack((positive_ix, positive_class_ids), dim=1)

        # Gather the masks (predicted and true) that contribute to loss
        y_true = target_masks[indices[:,0].data,:,:]
        y_pred = pred_masks[indices[:,0].data,indices[:,1].data,:,:]

        # Binary cross entropy
        loss = F.binary_cross_entropy(y_pred, y_true)
    else:
        loss = Variable(torch.FloatTensor([0]), requires_grad=False)
        if target_class_ids.is_cuda:
            loss = loss.cuda()

    return loss


def compute_losses(rpn_match, rpn_bbox, inputs):

    rpn_class_logits, rpn_pred_bbox, target_class_ids, \
        mrcnn_class_logits, target_deltas, mrcnn_bbox, target_mask, mrcnn_mask = \
        inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5], inputs[6], inputs[7]

    rpn_class_loss = compute_rpn_class_loss(rpn_match, rpn_class_logits)
    rpn_bbox_loss = compute_rpn_bbox_loss(rpn_bbox, rpn_match, rpn_pred_bbox)
    mrcnn_class_loss = compute_mrcnn_class_loss(target_class_ids, mrcnn_class_logits)
    mrcnn_bbox_loss = compute_mrcnn_bbox_loss(target_deltas, target_class_ids, mrcnn_bbox)
    mrcnn_mask_loss = compute_mrcnn_mask_loss(target_mask, target_class_ids, mrcnn_mask)

    outputs = [rpn_class_loss, rpn_bbox_loss, mrcnn_class_loss, mrcnn_bbox_loss, mrcnn_mask_loss]
    return sum(outputs), outputs


# TODO: the following is a long to-do list
def valid_epoch(model, datagenerator, steps):

    step, loss_sum = 0, 0

    for inputs in datagenerator:
        images = inputs[0]
        image_metas = inputs[1]
        rpn_match = inputs[2]
        rpn_bbox = inputs[3]
        gt_class_ids = inputs[4]
        gt_boxes = inputs[5]
        gt_masks = inputs[6]

        # image_metas as numpy array
        image_metas = image_metas.numpy()

        # Wrap in variables
        images = Variable(images, volatile=True)
        rpn_match = Variable(rpn_match, volatile=True)
        rpn_bbox = Variable(rpn_bbox, volatile=True)
        gt_class_ids = Variable(gt_class_ids, volatile=True)
        gt_boxes = Variable(gt_boxes, volatile=True)
        gt_masks = Variable(gt_masks, volatile=True)

        # To GPU
        if self.config.GPU_COUNT:
            images = images.cuda()
            rpn_match = rpn_match.cuda()
            rpn_bbox = rpn_bbox.cuda()
            gt_class_ids = gt_class_ids.cuda()
            gt_boxes = gt_boxes.cuda()
            gt_masks = gt_masks.cuda()

        # Run object detection
        rpn_class_logits, rpn_pred_bbox, target_class_ids, mrcnn_class_logits, \
            target_deltas, mrcnn_bbox, target_mask, mrcnn_mask = \
            self.predict([images, image_metas, gt_class_ids, gt_boxes, gt_masks], mode='training')

        if not target_class_ids.size():
            continue

        # Compute losses
        rpn_class_loss, rpn_bbox_loss, mrcnn_class_loss, mrcnn_bbox_loss, mrcnn_mask_loss = \
            compute_losses(rpn_match, rpn_bbox, rpn_class_logits, rpn_pred_bbox, target_class_ids,
                           mrcnn_class_logits, target_deltas, mrcnn_bbox, target_mask, mrcnn_mask)
        loss = rpn_class_loss + rpn_bbox_loss + mrcnn_class_loss + mrcnn_bbox_loss + mrcnn_mask_loss

        # Progress
        utils.printProgressBar(step + 1, steps, prefix="\t{}/{}".format(step + 1, steps),
                               suffix="Complete - loss: {:.5f} - rpn_class_loss: {:.5f} - rpn_bbox_loss: {:.5f} - "
                                      "mrcnn_class_loss: {:.5f} - mrcnn_bbox_loss: {:.5f} - "
                                      "mrcnn_mask_loss: {:.5f}".format(
                                   loss.data.cpu()[0],
                                   rpn_class_loss.data.cpu()[0], rpn_bbox_loss.data.cpu()[0],
                                   mrcnn_class_loss.data.cpu()[0], mrcnn_bbox_loss.data.cpu()[0],
                                   mrcnn_mask_loss.data.cpu()[0]), length=10)
        # Statistics
        loss_sum += loss.data.cpu()[0]/steps

        # Break after 'steps' steps
        if step == steps-1:
            break
        step += 1

    return loss_sum


############################################################
#  COCO Evaluation
############################################################
def evaluate_coco(model, dataset, coco_api, eval_type="bbox", limit=0, image_ids=None):
    """
        Runs official COCO evaluation.
        dataset:    A Dataset object with validation datasets
        eval_type:  "bbox" or "segm" for bounding box or segmentation evaluation
        limit:      the number of images to use for evaluation
    """
    # Pick COCO images from the dataset
    image_ids = image_ids or dataset.image_ids

    # Limit to a subset
    if limit:
        image_ids = image_ids[:limit]

    # Get corresponding COCO image IDs.
    coco_image_ids = [dataset.image_info[id]["id"] for id in image_ids]

    t_prediction = 0
    t_start = time.time()

    results = []
    for i, image_id in enumerate(image_ids):
        # Load image
        image = dataset.load_image(image_id)

        # Run detection
        t = time.time()
        res_raw = detect(model, [image])[0]
        t_prediction += (time.time() - t)

        # Convert results to COCO format
        image_results = build_coco_results(dataset, coco_image_ids[i:i + 1],
                                           res_raw["rois"], res_raw["class_ids"],
                                           res_raw["scores"], res_raw["masks"])
        results.extend(image_results)

        if i % 1000 == 0 or i == len(image_ids):
            print('eval progress (single gpu)\t{:4d}/{:4d} ...'.format(i, len(image_ids)))

    # Load results. This modifies results with additional attributes.
    coco_results = coco_api.loadRes(results)

    # Evaluate
    print('begin to evaluate ...')
    cocoEval = COCOeval(coco_api, coco_results, eval_type)
    cocoEval.params.imgIds = coco_image_ids
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    print("Prediction time: {}. Average {}/image".format(
        t_prediction, t_prediction / len(image_ids)))
    print("Total time: ", time.time() - t_start)


def detect(model, images):
    """
        'forward' method FOR EVALUATION ONLY.
        Runs the detection pipeline.
        images: List of images, potentially of different sizes.

        Returns a list of dicts, one dict per image. The dict contains:
            rois: [N, (y1, x1, y2, x2)] detection bounding boxes
            class_ids: [N] int class IDs
            scores: [N] float probability scores for the class IDs
            masks: [H, W, N] instance binary masks
    """

    # Mold inputs to format expected by the neural network
    molded_images, image_metas, windows = mold_inputs(model, images)

    # Convert images to torch tensor
    molded_images = torch.from_numpy(molded_images.transpose(0, 3, 1, 2)).float()
    molded_images = Variable(molded_images.cuda(), volatile=True)

    # Run object detection
    detections, mrcnn_mask = model([molded_images, image_metas], mode='inference')

    # Convert to numpy
    detections = detections.data.cpu().numpy()
    mrcnn_mask = mrcnn_mask.permute(0, 1, 3, 4, 2).data.cpu().numpy()

    # Process detections
    results = []
    for i, image in enumerate(images):
        final_rois, final_class_ids, final_scores, final_masks =\
            unmold_detections(detections[i], mrcnn_mask[i], image.shape, windows[i])
        results.append({
            "rois": final_rois,
            "class_ids": final_class_ids,
            "scores": final_scores,
            "masks": final_masks,
        })
    return results


def mold_inputs(model, images):
    """
        FOR EVALUATION ONLY.
        Takes a list of images and modifies them to the format expected as an input to the neural network.
        images: List of image matricies [height,width,depth]. Images can have different sizes.

        Returns 3 Numpy matricies:
            molded_images: [N, h, w, 3]. Images resized and normalized.
            image_metas: [N, length of meta datasets]. Details about each image.
            windows: [N, (y1, x1, y2, x2)]. The portion of the image that has the
            original image (padding excluded).
    """
    molded_images = []
    image_metas = []
    windows = []
    for image in images:
        # Resize image to fit the model expected size
        # TODO: move resizing to mold_image()
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
    # Pack into arrays
    molded_images = np.stack(molded_images)
    image_metas = np.stack(image_metas)
    windows = np.stack(windows)
    return molded_images, image_metas, windows


def unmold_detections(detections, mrcnn_mask, image_shape, window):
    """
        FOR EVALUATION ONLY.
        Reformats the detections of one image from the format of the neural
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


def build_coco_results(dataset, image_ids, rois, class_ids, scores, masks):
    """Arrange resutls to match COCO specs in http://cocodataset.org/#format
    """
    # If no results, return an empty list
    if rois is None:
        return []

    results = []
    for image_id in image_ids:
        # Loop through detections
        for i in range(rois.shape[0]):
            class_id = class_ids[i]
            score = scores[i]
            bbox = np.around(rois[i], 1)
            mask = masks[:, :, i]

            result = {
                "image_id": image_id,
                "category_id": dataset.get_source_class_id(class_id, "coco"),
                "bbox": [bbox[1], bbox[0], bbox[3] - bbox[1], bbox[2] - bbox[0]],
                "score": score,
                "segmentation": maskUtils.encode(np.asfortranarray(mask))
            }
            results.append(result)
    return results
