#!/usr/bin/env bash

# for single card eval
# python3.7 tools/eval.py -c ./ShuffleNet/ShuffleNetV2_x0_25.yaml

# for multi-cards eval
python3.7 -m paddle.distributed.launch --gpus="0,1,2,3" tools/eval.py -c ./ShuffleNet/ShuffleNetV2_x0_25.yaml
