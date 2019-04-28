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
4. Download pretrained ResNet-18 [model](https://download.pytorch.org/models/resnet18-5c106cde.pth) 
and put it under `${ROOT}/models/pytorch/imagenet/`

### Training with simulated data
To train with simulated data, run:
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
