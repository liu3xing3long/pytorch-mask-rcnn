import re
from lib.workflow import *
from lib.model import *
import torch
import torch.nn as nn
from torch.autograd import Variable


class MaskRCNN(nn.Module):
    def __init__(self, config):
        """
            config: A Sub-class of the Config class
            model_dir: Directory to save training results and trained weights
        """
        super(MaskRCNN, self).__init__()
        self.config = config
        self.loss_history = []
        self.val_loss_history = []

        self._build(config=config)
        self._initialize_weights()
        # self._set_log_dir()
        # self._epoch = 0
        # self._iter = 0

    @property
    def epoch(self):
        return self._epoch

    @epoch.setter
    def epoch(self, value):
        self._epoch = value

    @property
    def iter(self):
        return self._iter

    @iter.setter
    def iter(self, value):
        self._iter = value

    def _build(self, config):
        """Build Mask R-CNN architecture: fpn, rpn, classifier, mask"""

        # Image size must be dividable by 2 multiple times
        h, w = config.DATA.IMAGE_SHAPE[:2]
        if h / 2**6 != int(h / 2**6) or w / 2**6 != int(w / 2**6):
            raise Exception("Image size must be dividable by 2 at least 6 times "
                            "to avoid fractions when downscaling and upscaling."
                            "For example, use 256, 320, 384, 448, 512, ... etc. ")

        # Build the shared convolutional layers.
        # Bottom-up Layers
        # Returns a list of the last layers of each stage, 5 in total.
        # Don't create the thead (stage 5), so we pick the 4th item in the list.
        resnet = ResNet("resnet101", stage5=True)
        C1, C2, C3, C4, C5 = resnet.stages()

        # Top-down Layers
        # TODO (low): add assert to verify feature map sizes match what is in config
        self.fpn = FPN(C1, C2, C3, C4, C5, out_channels=256)

        # Generate Anchors (Tensor; do not assign cuda() here)
        self.priors = torch.from_numpy(
            generate_pyramid_priors(config.RPN.ANCHOR_SCALES, config.RPN.ANCHOR_RATIOS,
                                    config.MODEL.BACKBONE_SHAPES, config.MODEL.BACKBONE_STRIDES,
                                    config.RPN.ANCHOR_STRIDE)).float()

        # RPN
        self.rpn = RPN(len(config.RPN.ANCHOR_RATIOS), config.RPN.ANCHOR_STRIDE, 256)
        # FPN Classifier
        self.classifier = Classifier(256, config.MRCNN.POOL_SIZE,
                                     config.DATA.IMAGE_SHAPE, config.DATASET.NUM_CLASSES)
        # FPN Mask
        self.mask = Mask(256, config.MRCNN.MASK_POOL_SIZE, config.DATA.IMAGE_SHAPE, config.DATASET.NUM_CLASSES)

        # Fix batch norm layers
        def set_bn_fix(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1:
                for p in m.parameters():
                    p.requires_grad = False
        self.apply(set_bn_fix)

    def _initialize_weights(self):

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def set_trainable(self, layer_regex):
        """called in 'workflow.py'
        Sets model layers as trainable if their names match the given regular expression.
        """
        for param in self.named_parameters():
            layer_name = param[0]
            trainable = bool(re.fullmatch(layer_regex, layer_name))
            if not trainable:
                param[1].requires_grad = False

    @staticmethod
    def adjust_input_gt(*args):
        """zero-padding different number of GTs for each image within the batch"""
        gt_cls_ids = args[0]
        gt_boxes = args[1]
        gt_masks = args[2]
        gt_num = [x.shape[0] for x in gt_cls_ids]
        max_gt_num = max(gt_num)
        bs = len(gt_cls_ids)
        mask_shape = gt_masks[0].shape[1]

        GT_CLS_IDS = torch.zeros(bs, max_gt_num)
        GT_BOXES = torch.zeros(bs, max_gt_num, 4)
        GT_MASKS = torch.zeros(bs, max_gt_num, mask_shape, mask_shape)
        for i in range(bs):
            GT_CLS_IDS[i, :gt_num[i]] = torch.from_numpy(gt_cls_ids[i])
            GT_BOXES[i, :gt_num[i], :] = torch.from_numpy(gt_boxes[i]).float()
            GT_MASKS[i, :gt_num[i], :, :] = torch.from_numpy(gt_masks[i]).float()

        GT_CLS_IDS = Variable(GT_CLS_IDS.cuda(), requires_grad=False)
        GT_BOXES = Variable(GT_BOXES.cuda(), requires_grad=False)
        GT_MASKS = Variable(GT_MASKS.cuda(), requires_grad=False)

        return GT_CLS_IDS, GT_BOXES, GT_MASKS, gt_num

    def forward(self, input, mode):
        """forward function of the Mask-RCNN network"""
        molded_images = input[0]
        sample_per_gpu = molded_images.size(0)  # aka, actual batch size
        curr_coco_im_id = []

        # set model state
        if mode == 'inference':
            proposal_count = self.config.RPN.POST_NMS_ROIS_INFERENCE
            self.eval()
        elif mode == 'train':
            proposal_count = self.config.RPN.POST_NMS_ROIS_TRAINING
            self.train()
            # Set batchnorm always in eval mode during training
            def set_bn_eval(m):
                classname = m.__class__.__name__
                if classname.find('BatchNorm') != -1:
                    m.eval()
            self.apply(set_bn_eval)
        else:
            raise Exception('unknown phase')

        # Feature extraction
        [p2_out, p3_out, p4_out, p5_out, p6_out] = self.fpn(molded_images)

        # Note that P6 is used in RPN, but not in the classifier heads.
        _rpn_feature_maps = [p2_out, p3_out, p4_out, p5_out, p6_out]
        _mrcnn_feature_maps = [p2_out, p3_out, p4_out, p5_out]

        # Loop through pyramid layers
        layer_outputs = []  # list of lists
        for p in _rpn_feature_maps:
            layer_outputs.append(self.rpn(p))

        # Concatenate layer outputs
        # Convert from list of lists of level outputs to list of lists
        # of outputs across levels.
        # e.g. [[a1, b1, c1], [a2, b2, c2]] => [[a1, a2], [b1, b2], [c1, c2]]
        outputs = list(zip(*layer_outputs))
        outputs = [torch.cat(list(o), dim=1) for o in outputs]
        RPN_PRED_CLS_LOGITS, _rpn_class_score, RPN_PRED_BBOX = outputs

        # Generate proposals
        # Proposals are [batch, N, (y1, x1, y2, x2)] in normalized coordinates and zero padded.
        _proposals = proposal_layer([_rpn_class_score, RPN_PRED_BBOX],
                                    proposal_count=proposal_count,
                                    nms_threshold=self.config.RPN.NMS_THRESHOLD,
                                    priors=self.priors, config=self.config)
        # Normalize coordinates
        h, w = self.config.DATA.IMAGE_SHAPE[:2]
        scale = Variable(torch.from_numpy(np.array([h, w, h, w])).float(), requires_grad=False).cuda()

        if self.config.CTRL.PROFILE_ANALYSIS:
            curr_gpu_id = torch.cuda.current_device()
            curr_coco_im_id = input[-1][:, -1]
            print('\t[gpu {:d}] curr_coco_im_ids: {}'.format(curr_gpu_id, curr_coco_im_id.data.cpu().numpy()))
            print('\t[gpu {:d}] pass feature extraction'.format(curr_gpu_id))

        if mode == 'inference':
            # Network Heads
            # Proposal classifier and BBox regressor heads
            _, mrcnn_class, mrcnn_bbox = self.classifier(_mrcnn_feature_maps, _proposals)

            # Detections
            image_metas = input[1]  # (3, 90), Variable
            # output is [batch, num_detections (say 100), (y1, x1, y2, x2, class_id, score)] in image coordinates
            detections = detection_layer(_proposals, mrcnn_class, mrcnn_bbox, image_metas, self.config)

            # Convert boxes to normalized coordinates
            normalize_boxes = detections[:, :, :4] / scale
            # Create masks for detections
            mrcnn_mask = self.mask(_mrcnn_feature_maps, normalize_boxes)

            # shape: batch, num_detections, 81, 28, 28
            mrcnn_mask = mrcnn_mask.view(sample_per_gpu, -1,
                                         mrcnn_mask.size(1), mrcnn_mask.size(2), mrcnn_mask.size(3))

            return [detections, mrcnn_mask]

        elif mode == 'train':

            gt_class_ids, gt_boxes, gt_masks = input[1], input[2], input[3]

            # compute RPN Targets
            target_rpn_match, target_rpn_bbox = \
                prepare_rpn_target(self.priors, gt_class_ids, gt_boxes, self.config, curr_coco_im_id)
            if self.config.CTRL.PROFILE_ANALYSIS:
                print('\t[gpu {:d}] pass rpn_target generation'.format(curr_gpu_id))

            # _rois: N, TRAIN_ROIS_PER_IMAGE, 4; zero padded
            _rois, target_class_ids, target_deltas, target_mask = \
                prepare_det_target(_proposals.detach(), gt_class_ids, gt_boxes/scale, gt_masks, self.config)

            if self.config.CTRL.PROFILE_ANALYSIS:
                print('\t[gpu {:d}] pass pass det_target generation'.format(curr_gpu_id))
            if torch.sum(_rois).data[0] != 0:
                # classifier
                mrcnn_cls_logits, _, mrcnn_bbox = self.classifier(_mrcnn_feature_maps, _rois)
                # mask
                mrcnn_mask = self.mask(_mrcnn_feature_maps, _rois)
                # reshape output
                mrcnn_class_logits = mrcnn_cls_logits.view(sample_per_gpu, -1, mrcnn_cls_logits.size(1))
                mrcnn_bbox = mrcnn_bbox.view(sample_per_gpu, -1, mrcnn_bbox.size(1), mrcnn_bbox.size(2))
                mrcnn_mask = mrcnn_mask.view(sample_per_gpu, -1,
                                             mrcnn_mask.size(1), mrcnn_mask.size(2), mrcnn_mask.size(3))
            else:
                # if **ALL** samples within the batch has empty "_rois", skip the heads and output zero predictions.
                # this is really rare case. otherwise, pass the heads even some samples don't have _rois.
                num_rois, mask_sz, num_cls = self.config.ROIS.TRAIN_ROIS_PER_IMAGE, \
                                             self.config.MRCNN.MASK_SHAPE[0], self.config.DATASET.NUM_CLASSES
                mrcnn_class_logits = Variable(torch.zeros(sample_per_gpu, num_rois, num_cls).cuda())
                mrcnn_bbox = Variable(torch.zeros(sample_per_gpu, num_rois, num_cls, 4).cuda())
                mrcnn_mask = Variable(torch.zeros(sample_per_gpu, num_rois, num_cls, mask_sz, mask_sz).cuda())

            if self.config.CTRL.PROFILE_ANALYSIS:
                print('\t[gpu {:d}] pass mask and cls generation'.format(curr_gpu_id))
            # compute loss directly
            rpn_class_loss = compute_rpn_class_loss(target_rpn_match, RPN_PRED_CLS_LOGITS)
            rpn_bbox_loss = compute_rpn_bbox_loss(target_rpn_bbox, target_rpn_match, RPN_PRED_BBOX)
            mrcnn_class_loss = compute_mrcnn_class_loss(target_class_ids, mrcnn_class_logits)
            mrcnn_bbox_loss = compute_mrcnn_bbox_loss(target_deltas, target_class_ids, mrcnn_bbox)
            mrcnn_mask_loss = compute_mrcnn_mask_loss(target_mask, target_class_ids, mrcnn_mask)

            outputs = torch.stack((rpn_class_loss, rpn_bbox_loss,
                                   mrcnn_class_loss, mrcnn_bbox_loss, mrcnn_mask_loss), dim=1)
            if self.config.CTRL.PROFILE_ANALYSIS:
                print('\t[gpu {:d}] pass loss compute!'.format(curr_gpu_id))
            return outputs

            # return [target_rpn_match, RPN_PRED_CLS_LOGITS,
            #         target_rpn_bbox, RPN_PRED_BBOX,
            #         target_class_ids, mrcnn_class_logits,
            #         target_deltas, mrcnn_bbox,
            #         target_mask, mrcnn_mask]

