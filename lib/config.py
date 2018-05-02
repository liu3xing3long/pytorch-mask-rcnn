from tools.utils import *
from tools.collections import AttrDict


class Config(object):
    """Base configuration class. For custom configurations, create a
    sub-class that inherits from this one and override properties
    that need to be changed.
    """

    # ==================================
    MODEL = AttrDict()
    # Path to pretrained imagenet model
    MODEL.PRETRAIN_IMAGENET_MODEL = os.path.join('datasets/pretrain_model', "resnet50_imagenet.pth")
    # Path to pretrained weights file
    MODEL.PRETRAIN_COCO_MODEL = os.path.join('datasets/pretrain_model', 'mask_rcnn_coco.pth')
    MODEL.INIT_FILE_CHOICE = 'last'  # or file (xxx.pth)
    MODEL.INIT_MODEL = None   # set in 'utils.py'

    MODEL.BACKBONE = 'resnet101'  # todo: other structures (ssd, etc.)
    MODEL.BACKBONE_STRIDES = []
    MODEL.BACKBONE_SHAPES = []

    # ==================================
    DATASET = AttrDict()
    # Number of classification classes (including background)
    DATASET.NUM_CLASSES = 81
    DATASET.YEAR = '2014'
    DATASET.PATH = 'datasets/coco'

    # ==================================
    RPN = AttrDict()
    # Length of square anchor side in pixels
    RPN.ANCHOR_SCALES = (32, 64, 128, 256, 512)

    # Ratios of anchors at each cell (width/height)
    # A value of 1 represents a square anchor, and 0.5 is a wide anchor
    RPN.ANCHOR_RATIOS = [0.5, 1, 2]

    # Anchor stride
    # If 1 then anchors are created for each cell in the backbone feature map.
    # If 2, then anchors are created for every other cell, and so on (stride=2,3,4...).
    RPN.ANCHOR_STRIDE = 1

    # Non-max suppression threshold to filter RPN proposals.
    # You can reduce this during training to generate more proposals.
    RPN.NMS_THRESHOLD = 0.7

    # How many anchors per image to use for RPN training
    RPN.TRAIN_ANCHORS_PER_IMAGE = 256

    # ROIs kept after non-maximum suppression for RPN part
    RPN.PRE_NMS_LIMIT = 6000
    RPN.POST_NMS_ROIS_TRAINING = 2000
    RPN.POST_NMS_ROIS_INFERENCE = 1000

    RPN.TARGET_POS_THRES = .7
    RPN.TARGET_NEG_THRES = .3

    # ==================================
    MRCNN = AttrDict()
    # If enabled, resize instance masks to a smaller size to reduce
    # memory load. Recommended when using high-resolution images.
    MRCNN.USE_MINI_MASK = True
    MRCNN.MINI_MASK_SHAPE = (56, 56)  # (height, width) of the mini-mask
    # Pooled ROIs
    MRCNN.POOL_SIZE = 7         # cls/bbox stream
    MRCNN.MASK_POOL_SIZE = 14   # mask stream
    MRCNN.MASK_SHAPE = [28, 28]

    # ==================================
    DATA = AttrDict()
    # Input image resize
    # Images are resized such that the smallest side is >= IMAGE_MIN_DIM and
    # the longest side is <= IMAGE_MAX_DIM. In case both conditions can't
    # be satisfied together the IMAGE_MAX_DIM is enforced.
    DATA.IMAGE_MIN_DIM = 800
    DATA.IMAGE_MAX_DIM = 1024
    # If True, pad images with zeros such that they're (max_dim by max_dim)
    DATA.IMAGE_PADDING = True  # currently, the False option is not supported

    # Image mean (RGB)
    DATA.MEAN_PIXEL = np.array([123.7, 116.8, 103.9])

    # Maximum number of ground truth instances to use in one image
    DATA.MAX_GT_INSTANCES = 100

    # Bounding box refinement standard deviation for RPN and final detections.
    DATA.BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])
    DATA.IMAGE_SHAPE = []

    # ==================================
    ROIS = AttrDict()
    # Number of ROIs per image to feed to classifier/mask heads
    # The Mask RCNN paper uses 512 but often the RPN doesn't generate
    # enough positive proposals to fill this and keep a positive:negative
    # ratio of 1:3. You can increase the number of proposals by adjusting the RPN NMS threshold.
    ROIS.TRAIN_ROIS_PER_IMAGE = 200

    # Percent of positive ROIs used to train classifier/mask heads
    ROIS.ROI_POSITIVE_RATIO = 0.33

    ROIS.METHOD = 'roi_align'  # todo: regular roi_pooling

    # ==================================
    TEST = AttrDict()
    TEST.BATCH_SIZE = 0   # set in _set_value()
    # Max number of final detections
    TEST.DET_MAX_INSTANCES = 100
    # Minimum probability value to accept a detected instance
    # ROIs below this threshold are skipped
    TEST.DET_MIN_CONFIDENCE = 0
    # Non-maximum suppression threshold for detection
    TEST.DET_NMS_THRESHOLD = 0.3
    TEST.SAVE_IM = False

    # ==================================
    TRAIN = AttrDict()
    TRAIN.BATCH_SIZE = 6
    # Learning rate and momentum
    # The Mask RCNN paper uses lr=0.02, but on TensorFlow it causes
    # weights to explode. Likely due to differences in optimzer implementation.
    TRAIN.OPTIM_METHOD = 'sgd'
    TRAIN.INIT_LR = 0.01
    TRAIN.MOMENTUM = 0.9
    # Weight decay regularization
    TRAIN.WEIGHT_DECAY = 0.0001
    TRAIN.GAMMA = 0.1
    TRAIN.LR_POLICY = 'steps_with_decay'
    TRAIN.END2END = False
    # in epoch
    TRAIN.SCHEDULE = [6, 4, 3]
    TRAIN.LR_WARM_UP = False
    TRAIN.LR_WP_ITER = 500
    TRAIN.LR_WP_FACTOR = 1. / 3.

    TRAIN.CLIP_GRAD = True
    TRAIN.MAX_GRAD_NORM = 5.0

    # let bn learn and also apply the same weight decay when setting up optimizer
    TRAIN.BN_LEARN = False

    # evaluate mAP after each stage
    TRAIN.DO_VALIDATION = True
    TRAIN.SAVE_FREQ_WITHIN_EPOCH = 10

    # ==============================
    DEV = AttrDict()
    DEV.SWITCH = False
    DEV.EFFECTIVE_AFER_ITER = -1  # set to <= 0 if trained from the very first iter
    DEV.UPSAMPLE_FAC = 2.
    DEV.LOSS_CHOICE = 'l2'   # TODO (high, urgent) 'ot', 'kl', etc.
    DEV.LOSS_FAC = 0.5
    DEV.BUFFER_SIZE = 1000  # set to 1 if use all historic data
    DEV.FEAT_BRANCH_POOL_SIZE = 14

    # ==============================
    CTRL = AttrDict()
    CTRL.CONFIG_NAME = ''
    CTRL.PHASE = ''
    CTRL.DEBUG = None
    CTRL.QUICK_VERIFY = False   # train on minival and test also on minival

    CTRL.SHOW_INTERVAL = 50
    CTRL.USE_VISDOM = False
    CTRL.PROFILE_ANALYSIS = False  # show time for some pass

    # ==============================
    MISC = AttrDict()
    # the following will be set somewhere else
    MISC.LOG_FILE = None
    MISC.DET_RESULT_FILE = None
    MISC.SAVE_IMAGE_DIR = None
    MISC.RESULT_FOLDER = None
    MISC.DEVICE_ID = []
    MISC.GPU_COUNT = -1

    def display(self, log_file):
        """Display *final* configuration values."""
        print_log("Configurations:", file=log_file)
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                value = getattr(self, a)
                if isinstance(value, AttrDict):
                    print_log("{}:".format(a), log_file)
                    for _, key in enumerate(value):
                        print_log("\t{:30}\t\t{}".format(key, value[key]), log_file)
                else:
                    print_log("{}\t{}".format(a, value), log_file)
        print_log("\n", log_file)

    def _set_value(self):
        """Set values of computed attributes. Override all previous settings."""

        if self.CTRL.DEBUG:
            self.CTRL.SHOW_INTERVAL = 1
            self.DATA.IMAGE_MIN_DIM = 320
            self.DATA.IMAGE_MAX_DIM = 512
            self.CTRL.PROFILE_ANALYSIS = False

        # set MISC.RESULT_FOLDER, 'results/base_101/train (or inference)/'
        self.MISC.RESULT_FOLDER = os.path.join(
            'results', self.CTRL.CONFIG_NAME.lower(), self.CTRL.PHASE)
        if not os.path.exists(self.MISC.RESULT_FOLDER):
            os.makedirs(self.MISC.RESULT_FOLDER)

        self.TEST.BATCH_SIZE = 2 * self.TRAIN.BATCH_SIZE

        # MUST be left at the end
        # The strides of each layer of the FPN Pyramid.
        if self.MODEL.BACKBONE == 'resnet101':
            self.MODEL.BACKBONE_STRIDES = [4, 8, 16, 32, 64]
        else:
            raise Exception('unknown backbone structure')

        # Input image size
        self.DATA.IMAGE_SHAPE = np.array(
            [self.DATA.IMAGE_MAX_DIM, self.DATA.IMAGE_MAX_DIM, 3])

        # Compute backbone size from input image size
        self.MODEL.BACKBONE_SHAPES = np.array(
            [[int(math.ceil(self.DATA.IMAGE_SHAPE[0] / stride)),
              int(math.ceil(self.DATA.IMAGE_SHAPE[1] / stride))]
             for stride in self.MODEL.BACKBONE_STRIDES])


class CocoConfig(Config):
    """Configuration for training on MS COCO.
    Derives from the base Config class and overrides values specific to the COCO dataset.
    """

    def __init__(self, args):
        super(CocoConfig, self).__init__()

        self.CTRL.CONFIG_NAME = args.config_name
        self.CTRL.PHASE = args.phase
        self.CTRL.DEBUG = args.debug

        self.MISC.DEVICE_ID = [int(x) for x in args.device_id.split(',')]
        self.MISC.GPU_COUNT = len(self.MISC.DEVICE_ID)

        _ignore_yaml_or_list = False
        # ================ (CUSTOMIZED CONFIG) =========================
        if args.config_name == 'fuck':
            # debug mode on local pc
            self.DEV.SWITCH = True
            self.DEV.BUFFER_SIZE = 1
            self.CTRL.QUICK_VERIFY = True
            _ignore_yaml_or_list = True

        elif args.config_name == 'base_101':
            self.MODEL.INIT_FILE_CHOICE = 'coco_pretrain'
            self.TRAIN.BATCH_SIZE = 16
            self.CTRL.PROFILE_ANALYSIS = False
            _ignore_yaml_or_list = True

        elif args.config_name == 'base_102':
            self.MODEL.INIT_FILE_CHOICE = 'imagenet_pretrain'
            self.CTRL.BATCH_SIZE = 16
            self.CTRL.PROFILE_ANALYSIS = False
            self.TEST.SAVE_IM = False
            _ignore_yaml_or_list = True

        elif args.config_name is None:
            if args.config_file is None:
                print('WARNING: No config file and config name! use default setting.'
                      'set config_name=default')
                self.CTRL.CONFIG_NAME = 'default'
            else:
                print('no config name but luckily you got config file ...')
        else:
            print('WARNING: unknown config name!!! use default setting.')
        # ================ (CUSTOMIZED CONFIG END) ======================

        # Optional
        if args.config_file is not None and not _ignore_yaml_or_list:
            print('Find .yaml file; use yaml name as CONFIG_NAME')
            self.CTRL.CONFIG_NAME = os.path.basename(args.config_file).replace('.yaml', '')
            merge_cfg_from_file(args.config_file, self)

        if len(args.opts) != 0 and not _ignore_yaml_or_list:
            print('Update configuration from terminal inputs ...')
            merge_cfg_from_list(args.opts, self)

        self._set_value()
