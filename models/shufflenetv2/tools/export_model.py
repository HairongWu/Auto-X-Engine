# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../')))

import paddle
import paddle.nn as nn

from ppcls.utils import config
from ppcls.engine.engine import Engine

from paddlelite.lite import *
import numpy as np

if __name__ == "__main__":
    args = config.parse_args()
    config = config.get_config(
        args.config, overrides=args.override, show=False)
    if config["Arch"].get("use_sync_bn", False):
        config["Arch"]["use_sync_bn"] = False
    engine = Engine(config, mode="export")
    engine.export()

    config = CxxConfig()
    config.set_model_dir(args.save_inference_dir)
    places = [Place(TargetType.X86, PrecisionType.FP32), Place(TargetType.Host, PrecisionType.FP32)]
    config.set_valid_places(places)
    predictor = create_paddle_predictor(config)
    predictor.save_optimized_model(args.save_inference_dir)


    input_tensor = predictor.get_input(0)
    input_tensor.from_numpy(np.ones([1,3, 192, 192]).astype('float32'))
    input_tensor2 = predictor.get_input(1)
    input_tensor2.from_numpy(np.ones([1, 2]).astype('float32'))

    predictor.run()

    output_tensor = predictor.get_output(0)
    print(output_tensor.numpy())
