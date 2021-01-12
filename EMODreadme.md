# Detecting Marine Organisms via Joint Attention-Relation Learning for Marine Video Surveillance
We release the code of our paper.

## Introduction

We propose a novel Attention-Relation (AR) module to explore joint Attention-Relation in CNNs for marine organism detection. We then design an Efficient Marine Organism Detector (EMOD) for high-resolution marine video surveillance to detect organisms and surveil marine environments in a real-time and fast fashion.

<div align="center">
  <img src="./EMOD framework.jpg" height="460px"/> 
</div>

<div align="center">
  <img src="./AR-method.png" height="340px"/>
</div>

This code is based on the [mmdetection](https://github.com/open-mmlab/mmdetection) codebase. 

## Requirements

- Linux or macOS (Windows is in experimental support)
- Python 3.6+
- PyTorch 1.3+
- CUDA 9.2+ (If you build PyTorch from source, CUDA 9.0 is also compatible)
- GCC 5+
- MMCV

## Datasets

#### MOUSS, MBARI and HabCam
- Download the dataset and annotations from [dataset provider](https://www.viametoolkit.org/). 
- Put all annotation json files in the same folder, and set `ann_file` to the path. Set `img_prefix` to be the path to the folder containing data.
- Make train and validation splits via [data/make_splits.py](data/make_splits.py)

## Running

- To train and test Faster-RCNN-ResNet50 from scratch on MOUSS0. You can run the scripts.

  ```
  CUDA_VISIBLE_DEVICES=0,1 PORT=29500 ./tools/dist_train.sh configs/faster_rcnn/faster_rcnn_x101_fpn_mouss0_sp1.py 2
  CUDA_VISIBLE_DEVICES=2,3 PORT=29501 ./tools/dist_train.sh configs/faster_rcnn/faster_rcnn_x101_fpn_mouss0_sp2.py 2
  ```
  
## Model Zoo

### Pretrain models

#### Cascade R-CNN 

Please refer to [Cascade R-CNN](https://github.com/open-mmlab/mmdetection/blob/master/configs/cascade_rcnn) for details.

#### RetinaNet

Please refer to [RetinaNet](https://github.com/open-mmlab/mmdetection/blob/master/configs/retinanet) for details.

#### Faster R-CNN

Please refer to [Faster R-CNN](https://github.com/open-mmlab/mmdetection/blob/master/configs/faster_rcnn) for details.

### YOLOv3

Please refer to [YOLOv3](https://github.com/open-mmlab/mmdetection/blob/master/configs/yolo) for details.

## Acknowledgement
We really appreciate the contributors of following codebases.

- [open-mmlab/mmdetection](https://github.com/open-mmlab/mmdetection)

