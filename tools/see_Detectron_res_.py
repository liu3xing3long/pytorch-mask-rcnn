import os
import pickle
import torch


# file_root = os.path.join('/home/hongyang/project/Detectron/results/e2e_mask_rcnn_new/test/coco2014_minival/generalized_rcnn')
# file = os.path.join(file_root, 'detections.pkl')
#
# with open(file, 'rb') as f:
#     det = pickle.load(f, encoding='latin1')   # or 'bytes'


our_file = os.path.join('results/all_new_2/inference', 'detection_result_0020_iter_42.pth')
dets = torch.load(our_file)
dets = dets['det_result']


a = 1