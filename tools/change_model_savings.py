import torch
import os

# in terminal, execute python this_file_name.py, the following path is right;
# however, in MacPycharm, it sees tools folder as root
base_name = 'results/meta_101_quick_3_l1_sig_multi/train'
file_name = 'mask_rcnn_ep_0006_iter_001238.pth'
# old file
model_path = os.path.join(base_name, file_name)
# load model
checkpoints = torch.load(model_path)

# CHANGE YOUR NEED HERE
checkpoints['iter'] -= 1


# Do **NOT** change the following
_ep = checkpoints['epoch']
_iter = checkpoints['iter']
# new file
model_file = os.path.join(base_name, 'mask_rcnn_ep_{:04d}_iter_{:06d}.pth'.format(_ep, _iter))
print('saving file: {}'.format(model_file))
torch.save(checkpoints, model_file)
if model_path == model_file:
    print('old name and new name is the same! will not delete old file!')
else:
    print('removing old file: {}'.format(model_path))
    os.remove(model_path)




