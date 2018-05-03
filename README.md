# Mask-RCNN
### A PyTorch Implementation (multi-gpu), adaption from a public [repository](https://github.com/multimodallearning/pytorch-mask-rcnn).





```Shell
This is a Pytorch implementation of [Mask R-CNN](https://arxiv.org/abs/1703.06870) that is in large parts
based on Matterport's [Mask_RCNN](https://github.com/matterport/Mask_RCNN). Matterport's repository
is an implementation on Keras and TensorFlow. The following parts of the README are excerpts from the Matterport README.
Details on the requirements, training on MS COCO and detection results for this repository can be found
at the end of the document. The Mask R-CNN model generates bounding boxes and segmentation masks
for each instance of an object in the image. It's based on Feature Pyramid Network (FPN) and a ResNet101 backbone.
```





## Installation
1. Clone this repository.

        git clone --recursive https://github.com/hli2020/pytorch-mask-rcnn.git

    
2. We use functions from two more repositories that need to be build with the right `--arch` option for cuda support.
The two functions are Non-Maximum Suppression from ruotianluo's [pytorch-faster-rcnn](https://github.com/ruotianluo/pytorch-faster-rcnn)
repository and longcw's [RoiAlign](https://github.com/longcw/RoIAlign.pytorch).

    | GPU | arch |
    | --- | --- |
    | TitanX | sm_52 |
    | GTX 960M | sm_50 |
    | GTX 1070 | sm_61 |
    | GTX 1080 (Ti) | sm_61 |

        sh setup.sh

3. As we use the [COCO dataset](http://cocodataset.org/#home),
install the [Python COCO API](https://github.com/cocodataset/cocoapi) and
create a symlink.

        ln -s /path/to/coco datasets/coco

4. Download the pretrained models on COCO and ImageNet from
[Google Drive](https://drive.google.com/open?id=1LXUgC2IZUYNEoXr05tdqyKFZY0pZyPDc).

## Demo

To test your installation simply run the demo with

    python demo.py


![](assets/park.png)

## Training on COCO
See the `script` folder to get a sense of training/evaluation commands in terminal.

The training schedule, learning rate, and other parameters can be set in the `class`
object of `CocoConfig` in `lib/config.py`.

## Results

COCO results for bounding box and segmentation are reported based on training
with the default configuration and backbone initialized with pretrained
**ImageNet** weights. Used metric is AP on IoU=0.50:0.95.

|    | from scratch | converted from keras | Matterport's Mask_RCNN | Mask R-CNN paper |
| --- | --- | --- | --- | --- |
| bbox | TODO | 0.347 | 0.347 | 0.382 |
| segm | TODO | 0.296 | 0.296 | 0.354 |


#### Annoying warning
``~/anaconda3/lib/python3.6/site-packages/scipy/ndimage/interpolation.py:616``

Also install the `future` package via conda:

``conda install -c anaconda future``



