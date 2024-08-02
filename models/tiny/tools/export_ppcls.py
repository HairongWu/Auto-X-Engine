import os
import sys
from functools import reduce
from operator import mul
import numpy as np
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../')))

from ppcls.utils import config
from ppcls.engine.engine import Engine

from paddlelite.lite import *
from naivebuffer import *
from nb2c import *

if __name__ == "__main__":
    args = config.parse_args()
    config = config.get_config(
        args.config, overrides=args.override, show=False)
    print(args)
    if config["Arch"].get("use_sync_bn", False):
        config["Arch"]["use_sync_bn"] = False
    engine = Engine(config, mode="export")
    engine.export()

    config2 = CxxConfig()
    config2.set_model_dir(config.Global.save_inference_dir)
    places = [Place(TargetType.X86, PrecisionType.FP32), Place(TargetType.Host, PrecisionType.FP32)]
    config2.set_valid_places(places)
    predictor = create_paddle_predictor(config2)
    predictor.save_optimized_model(os.path.join(config.Global.save_inference_dir, "model"))


    input_tensor = predictor.get_input(0)
    input_tensor.from_numpy(np.ones([1,3, 224, 224]).astype('float32'))

    predictor.run()

    output_tensor = predictor.get_output(0)
    print(output_tensor.numpy())
    print(output_tensor.numpy().shape)

    ops, dim_dict, weights_dict = read_nb(os.path.join(config.Global.save_inference_dir, "model.nb"))

    nb2c(config.Global.save_inference_dir, ops, weights_dict, dim_dict)