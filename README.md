# Source Detection for Neuron-tree Reconstruction

## Introduction
Pytorch code for training and testing source detection models for neuron-tree reconstruction.


## Overview
- `source_detection/`: includes training and validation scripts.
- `lib/`: contains data preparation, model definition, and some utility functions.
- `experiments/`: contains `*.yaml` configuration files to run experiments.


## Requirements
The code is developed using python 3.7.2 on Ubuntu 18.04.1. NVIDIA GPUs ared needed to train and test. 
See [`requirements.txt`](requirements.txt) for other dependencies.

## Quick start
### Installation
1. Install pytorch >= v1.0.0 following [official instructions](https://pytorch.org/).
2. Clone this repo, and we will call the directory that you cloned as `${ROOT}`
3. Install dependencies.
   ```
   pip install -r requirements.txt
   ```
4. Download pretrained neuron-detection model [model](https://drive.google.com/file/d/1iX1oE3bhKzuAHLi0MsPXeNzoqRm4nfZe/view?usp=sharing), and put it under `${ROOT}/models/neuron/`

### Training with simulated data (optional)
1. Download ResNet-18 model pretrained on ImageNet [model](https://download.pytorch.org/models/resnet18-5c106cde.pth) 
and put it under `${ROOT}/models/pytorch/imagenet/`
2. To train with simulated data, run:
```
CUDA_VISIBLE_DEVICES=$GPU_ID python source_detection/train.py --cfg experiments/simulated/128x128_d256x3_adam_lr1e-3.yaml
```
Model checkpoints and logs will be saved into `output` folder while tensorboard logs will be saved into `log` folder.

### Testing with simulated data
To test with simulated data after training, run:
```
CUDA_VISIBLE_DEVICES=$GPU_ID python source_detection/validate.py --cfg experiments/simulated/128x128_d256x3_adam_lr1e-3.yaml
```
Tensorboard logs will be saved into `log` folder.

### Testing with real data
You need to put the real data (e.g. srep31332-s1.mat) into ./data/real_dataset/ folder first. To test with real data after training, run:
```
CUDA_VISIBLE_DEVICES=$GPU_ID python source_detection/test.py --cfg experiments/real_data/128x128_d256x3_adam_lr1e-3.yaml
```
Tensorboard logs will be saved into `log` folder. Modify TEST_SET entry (for different input file names) or X_MIN, Y_MIN, X_MAX, Y_MAX entries (for different region in the video) in the .yaml file if you need to.
To see Tensorboard logs with images from all time-steps, run:
```
tensorboard --logdir=$YOUR_LOG_DIR --samples_per_plugin images=0
```
