import re
from lib.model import *
from lib.layers import *
import torch
import torch.nn as nn
from torch.autograd import Variable


class MaskRCNN(nn.Module):
    def __init__(self, config, model_dir):
        """
            config: A Sub-class of the Config class
            model_dir: Directory to save training results and trained weights
        """
        super(MaskRCNN, self).__init__()
        self.config = config
        self.model_dir = model_dir
        self.loss_history = []
        self.val_loss_history = []

        self._build(config=config)
        self._initialize_weights()
        self._set_log_dir()

    def _build(self, config):
        """Build Mask R-CNN architecture: fpn, rpn, classifier, mask"""

        # Image size must be dividable by 2 multiple times
        h, w = config.IMAGE_SHAPE[:2]
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
        # TODO: add assert to verify feature map sizes match what is in config
        self.fpn = FPN(C1, C2, C3, C4, C5, out_channels=256)

        # Generate Anchors
        self.anchors = torch.from_numpy(
            utils.generate_pyramid_anchors(config.RPN_ANCHOR_SCALES,
                                           config.RPN_ANCHOR_RATIOS,
                                           config.BACKBONE_SHAPES,
                                           config.BACKBONE_STRIDES,
                                           config.RPN_ANCHOR_STRIDE)).float()

        # RPN
        self.rpn = RPN(len(config.RPN_ANCHOR_RATIOS), config.RPN_ANCHOR_STRIDE, 256)

        # FPN Classifier
        self.classifier = Classifier(256, config.POOL_SIZE, config.IMAGE_SHAPE, config.NUM_CLASSES)

        # FPN Mask
        self.mask = Mask(256, config.MASK_POOL_SIZE, config.IMAGE_SHAPE, config.NUM_CLASSES)

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

    def _set_log_dir(self):

        # Setup directory
        # e.g., results/hyli_default/train(or inference)/
        self.log_dir = os.path.join(self.model_dir, self.config.NAME.lower(), self.config.PHASE)

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        # Path to save after each epoch; used for training
        self.checkpoint_path = os.path.join(self.log_dir, "mask_rcnn_*epoch*.pth")
        self.checkpoint_path = self.checkpoint_path.replace("*epoch*", "{:04d}")

    def load_weights(self, filepath):
        """called in model.py"""
        if filepath is not None:
            if os.path.exists(filepath):
                # TODO: find start_iter within epoch
                self.load_state_dict(torch.load(filepath))
                try:
                    self.start_epoch
                    self.start_iter
                except:
                    self.start_epoch, self.start_iter = 0, 0

                self.epoch = self.start_epoch
            else:
                raise Exception("Weight file not found ...")

    def set_trainable(self, layer_regex):
        """called in 'model.py'
        Sets model layers as trainable if their names match the given regular expression.
        """
        for param in self.named_parameters():
            layer_name = param[0]
            trainable = bool(re.fullmatch(layer_regex, layer_name))
            if not trainable:
                param[1].requires_grad = False

    def forward(self, input, mode):
        """forward function of the Mask-RCNN network"""
        curr_gpu_id = torch.cuda.current_device()
        molded_images = input[0]
        # sample_per_gpu = int(self.config.BATCH_SIZE / self.config.GPU_COUNT)
        sample_per_gpu = molded_images.size(0)  # aka, batch size
        # if self.config.DEBUG:
        #     print('forward on gpu {:d} now...'.format(curr_gpu_id))
        if mode == 'inference':
            self.eval()
        elif mode == 'training':
            self.train()
            # Set batchnorm always in eval mode during training
            def set_bn_eval(m):
                classname = m.__class__.__name__
                if classname.find('BatchNorm') != -1:
                    m.eval()
            self.apply(set_bn_eval)

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
        rpn_class_logits, _rpn_class_score, rpn_pred_bbox = outputs

        # Generate proposals
        # Proposals are [batch, N, (y1, x1, y2, x2)] in normalized coordinates and zero padded.
        proposal_count = self.config.POST_NMS_ROIS_TRAINING if mode == "training" \
            else self.config.POST_NMS_ROIS_INFERENCE
        _proposals = proposal_layer([_rpn_class_score, rpn_pred_bbox],
                                    proposal_count=proposal_count,
                                    nms_threshold=self.config.RPN_NMS_THRESHOLD,
                                    anchors=self.anchors,
                                    config=self.config)
        # Normalize coordinates
        h, w = self.config.IMAGE_SHAPE[:2]
        scale = Variable(torch.from_numpy(np.array([h, w, h, w])).float(), requires_grad=False).cuda()

        if mode == 'inference':
            # Network Heads
            # Proposal classifier and BBox regressor heads
            _, mrcnn_class, mrcnn_bbox = self.classifier(_mrcnn_feature_maps, _proposals)

            # Detections
            start_sample_ind = curr_gpu_id*sample_per_gpu
            end_sample_ind = start_sample_ind+sample_per_gpu
            image_metas = input[1][start_sample_ind:end_sample_ind]  # (3, 89), ndarray
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
            gt_boxes = gt_boxes / scale

            _rois, target_class_ids, target_deltas, target_mask, volatiles = \
                prepare_det_target(_proposals, gt_class_ids, gt_boxes, gt_masks, self.config)

            if torch.sum(_rois).data[0] != 0:
                # Proposal classifier and BBox regressor heads
                mrcnn_cls_logits, _, mrcnn_bbox = self.classifier(_mrcnn_feature_maps, _rois)
                # Create masks for detections
                mrcnn_mask = self.mask(_mrcnn_feature_maps, _rois)

                mrcnn_class_logits = mrcnn_cls_logits.view(sample_per_gpu, -1, mrcnn_cls_logits.size(1))
                mrcnn_bbox = mrcnn_bbox.view(sample_per_gpu, -1, mrcnn_bbox.size(1), mrcnn_bbox.size(2))
                mrcnn_mask = mrcnn_mask.view(sample_per_gpu, -1,
                                             mrcnn_mask.size(1), mrcnn_mask.size(2), mrcnn_mask.size(3))
            else:
                [mrcnn_class_logits, mrcnn_bbox, mrcnn_mask] = volatiles

            output = [rpn_class_logits, rpn_pred_bbox,
                      target_class_ids, mrcnn_class_logits,
                      target_deltas, mrcnn_bbox,
                      target_mask, mrcnn_mask]
            # if self.config.DEBUG:
            #     for ind, out in enumerate(output):
            #         print('output {:d}, on gpu {:d}'.format(ind, out.get_device()))
            #     print('curr forward done!')
            return output

    def adjust_input_gt(self, *args):
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