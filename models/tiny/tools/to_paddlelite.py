# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "..")))

from paddlelite.lite import *
from naivebuffer import *
from nb2c import *


def main():
    out_path = './pretrain/Multilingual_PP-OCRv3_det_infer/'
    config2 = CxxConfig()
    config2.set_model_dir(out_path)
    places = [Place(TargetType.X86, PrecisionType.FP32), Place(TargetType.Host, PrecisionType.FP32)]
    config2.set_valid_places(places)
    predictor = create_paddle_predictor(config2)
    predictor.save_optimized_model(os.path.join(out_path,"model"))


    input_tensor = predictor.get_input(0)
    input_tensor.from_numpy(np.ones([1,3, 960, 960]).astype('float32'))

    predictor.run()

    output_tensor = predictor.get_output(0)
    print(output_tensor.numpy())

    ops, dim_dict, weights_dict = read_nb(os.path.join(out_path, "model.nb"))

    nb2c(out_path, ops, weights_dict, dim_dict)

if __name__ == "__main__":
    main()
