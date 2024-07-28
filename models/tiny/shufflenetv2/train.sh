#!/usr/bin/env bash

# for single card train
# python3.7 tools/train.py -c ./ShuffleNet/ShuffleNetV2_x0_25.yaml

# for multi-cards train
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3.7 -m paddle.distributed.launch --gpus="0,1,2,3" tools/train.py -c ./ShuffleNet/ShuffleNetV2_x0_25.yaml
