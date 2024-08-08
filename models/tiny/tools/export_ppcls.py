import os
import sys
from functools import reduce
from operator import mul
import numpy as np
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../')))
import cv2
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

    with open('./Ball.jpg', 'rb') as f:
        im_read = f.read()
    data = np.frombuffer(im_read, dtype='uint8')
    im = cv2.imdecode(data, 1)  # BGR mode, but need RGB mode
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    origin_shape = im.shape[:2]
    im_scale_y = 224 / float(origin_shape[0])
    im_scale_x = 224 / float(origin_shape[1])
    img = cv2.resize(
                im,
                None,
                None,
                fx=im_scale_x,
                fy=im_scale_y,
                interpolation=cv2.INTER_LINEAR)

    input_tensor = predictor.get_input(0)
    input_tensor.from_numpy(np.ones([1,3, 224, 224]).astype('float32'))

    predictor.run()

    output_tensor = predictor.get_output(0)
    print(output_tensor.numpy())
    print(output_tensor.numpy().shape)

    ops, dim_dict, weights_dict = read_nb(os.path.join(config.Global.save_inference_dir, "model.nb"))

    nb2c(config.Global.save_inference_dir, ops, weights_dict, dim_dict)