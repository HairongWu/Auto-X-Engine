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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

# add python path of PaddleDetection to sys.path
parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 2)))
sys.path.insert(0, parent_path)

# ignore warning log
import warnings
warnings.filterwarnings('ignore')

import paddle
from ppdet.core.workspace import load_config, merge_config
from ppdet.utils.check import check_gpu, check_version, check_config
from ppdet.utils.cli import ArgsParser
from ppdet.engine import Trainer
from ppdet.slim import build_slim_model

from paddlelite.lite import *
from naivebuffer import *

from ppdet.utils.logger import setup_logger
logger = setup_logger('export_model')


def parse_args():
    parser = ArgsParser()
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output_inference",
        help="Directory for storing the output model files.")
    parser.add_argument(
        "--export_serving_model",
        type=bool,
        default=False,
        help="Whether to export serving model or not.")
    parser.add_argument(
        "--slim_config",
        default=None,
        type=str,
        help="Configuration file of slim method.")
    args = parser.parse_args()
    return args


def run(FLAGS, cfg):
    # build detector
    trainer = Trainer(cfg, mode='test')

    # load weights
    if cfg.architecture in ['DeepSORT', 'ByteTrack']:
        trainer.load_weights_sde(cfg.det_weights, cfg.reid_weights)
    else:
        trainer.load_weights(cfg.weights)

    # export model
    trainer.export(FLAGS.output_dir)

    if FLAGS.export_serving_model:
        from paddle_serving_client.io import inference_model_to_serving
        model_name = os.path.splitext(os.path.split(cfg.filename)[-1])[0]

        inference_model_to_serving(
            dirname="{}/{}".format(FLAGS.output_dir, model_name),
            serving_server="{}/{}/serving_server".format(FLAGS.output_dir,
                                                         model_name),
            serving_client="{}/{}/serving_client".format(FLAGS.output_dir,
                                                         model_name),
            model_filename="model.pdmodel",
            params_filename="model.pdiparams")

    config2 = CxxConfig()
    config2.set_model_dir(FLAGS.output_dir)
    places = [Place(TargetType.X86, PrecisionType.FP32), Place(TargetType.Host, PrecisionType.FP32)]
    config2.set_valid_places(places)
    predictor = create_paddle_predictor(config2)
    predictor.save_optimized_model(FLAGS.output_dir+"model")


    input_tensor = predictor.get_input(0)
    input_tensor.from_numpy(np.ones([1,3, 320, 320]).astype('float32'))
    input_tensor2 = predictor.get_input(1)
    input_tensor2.from_numpy(np.ones([1, 2]).astype('float32'))

    predictor.run()

    output_tensor = predictor.get_output(0)
    print(output_tensor.numpy())

    ops, dim_dict, weights_dict = read_nb(FLAGS.output_dir+"model.nb")

    op_mapping = {'conv2d':'autox_conv2d','depthwise_conv2d':'autox_conv2d','pool2d':'autox_pool2d',
                  'concat':'autox_concat','transpose2':'autox_transpose2','split':'autox_split','matmul_v2':'autox_matmul_v2',
                  'elementwise_add':'autox_elementwise_add','softmax':'autox_softmax'}
    attrs = ['act_type','dilations','groups','paddings','strides','adaptive','ksize','pooling_type','axis','start_axis','stop_axis',
             'trans_x','trans_y']
    ignore_layers = ['feed','shape','slice','fill_constant','reshape2','flatten_contiguous_range', 'fetch']
    
    fw = open(FLAGS.output_dir+"model.bin", "wb")
    fm = open(FLAGS.output_dir+"model.h", "w")
    index = 0
    fm.write('void picodet_xs_320_coco_lcnet(const uint8_t *image, const uint16_t ssize_h, const uint16_t ssize_w, const float *weights, uint32_t *Out)\n{\n')
    for op in ops:
        if op.type in ignore_layers:
            continue

        for i in op.inputs:
            for arg in i.arguments:
                if arg in weights_dict:
                    for f in weights_dict[arg]:
                        fw.write(struct.pack('f',f))

        operator = ''
        
        for i in op.outputs:
            if 'XShape' == i.parameter:
                continue
            for arg in i.arguments:
                operator = operator + '\tfloat *'
                operator = operator + arg.replace('.','_')
                operator = operator + " = (float *)calloc(%d*sizeof(float));\n"%abs(reduce(mul, dim_dict[arg]))
        operator = operator +"\t"+ op_mapping[op.type]
        operator = operator + "("
        
        inputs = []
        weights = []
        outputs = []
        inputs_dim = []
        weights_dim = []
        outputs_dim = []
        attributes = {}
        for i in op.inputs:
            for arg in i.arguments:
                if arg in weights_dict:
                    weights.append("weights + " + str(index))
                    weights_dim.append('{'+', '.join([str(abs(i)) for i in dim_dict[arg]])+'}')
                    index = index + abs(reduce(mul, dim_dict[arg]))
                else:
                    inputs.append(arg.replace('.','_'))
                    inputs_dim.append('{'+', '.join([str(abs(i)) for i in dim_dict[arg]])+'}')
        for i in op.outputs:
             if 'XShape' == i.parameter:
                continue
             for arg in i.arguments:
                outputs.append(arg.replace('.','_'))
                outputs_dim.append('{'+', '.join([str(abs(i)) for i in dim_dict[arg]])+'}')

        for i in op.attrs:
            if i.name in attrs:
                if i.type == AttributeType.INT:
                    attributes[i.name] = str(i.i)
                elif i.type == AttributeType.FLOAT or i.type == AttributeType.FLOAT64:
                    attributes[i.name] = str(i.f)
                elif i.type == AttributeType.STRING:
                    attributes[i.name] = "\"" + i.s + "\""
                elif i.type == AttributeType.INTS:
                    attributes[i.name] = ', '.join([str(j) for j in i.ints])
                elif i.type == AttributeType.FLOATS:
                    attributes[i.name] = ', '.join([str(j) for j in i.floats])
                elif i.type == AttributeType.BOOLEAN:
                    if i.b:
                        attributes[i.name] = str(1)
                    else:
                        attributes[i.name] = str(0)
                else:
                    print(i.type)
                    break
        
        free_str = ''
        if op.type == "conv2d" or op.type == "depthwise_conv2d":
            for i in inputs:
                operator = operator + i
                operator = operator + ", "
                free_str = free_str + '\tfree(%s);\n'%(i)
            for i in outputs:
                operator = operator + i
                operator = operator + ", "
            for i in weights:
                operator = operator + i
                operator = operator + ", "
            for i in inputs_dim:
                operator = operator + i
                operator = operator + ", "
            for i in outputs_dim:
                operator = operator + i
                operator = operator + ", "

            operator = operator + weights_dim[1]
            operator = operator + ", "

            operator = operator + attributes['groups']
            operator = operator + ", "
            operator = operator + attributes['paddings']
            operator = operator + ", "
            operator = operator + attributes['strides']
            operator = operator + ", "
            operator = operator + attributes['dilations']
            operator = operator + ", "

            if 'act_type' in attributes:
                if 'relu' == ACTIVATION.kRelu:
                    operator = operator + str(ACTIVATION.kRelu)
            else:
                operator = operator + str(ACTIVATION.kIndentity)
            operator = operator + ", "
        elif op.type == "pool2d":
            for i in inputs:
                operator = operator + i
                operator = operator + ", "
                free_str = free_str + '\tfree(%s);\n'%(i)
            for i in outputs:
                operator = operator + i
                operator = operator + ", "
            for i in inputs_dim:
                operator = operator + i
                operator = operator + ", "
            for i in outputs_dim:
                operator = operator + i
                operator = operator + ", "

            operator = operator + attributes['ksize']
            operator = operator + ", "
            operator = operator + attributes['strides']
            operator = operator + ", "
            operator = operator + attributes['paddings']
            operator = operator + ", "
            operator = operator + attributes['adaptive']
            operator = operator + ", "
            if attributes['pooling_type'] == "avg":
                operator = operator + str(1)
            else:
                operator = operator + str(0)
            operator = operator + ", "
        elif op.type == "transpose2":
            for i in inputs:
                operator = operator + i
                operator = operator + ", "
                free_str = free_str + '\tfree(%s);\n'%(i)
            for i in outputs:
                operator = operator + i
                operator = operator + ", "
            for i in inputs_dim:
                operator = operator + i
                operator = operator + ", "
            for i in outputs_dim:
                operator = operator + i
                operator = operator + ", "

            operator = operator +"{"+ attributes['axis']+"}"
            operator = operator + ", "
            operator = operator + str(4)
            operator = operator + ", "
        elif op.type == "softmax":
            for i in inputs:
                operator = operator + i
                operator = operator + ", "
            
            for i in inputs_dim:
                operator = operator + i
                operator = operator + ", "
            
            for key in attributes:
                operator = operator + attributes[key]
                operator = operator + ", "
        else:
            for i in inputs:
                operator = operator + i
                operator = operator + ", "
                free_str = free_str + '\tfree(%s);\n'%(i)
            for i in outputs:
                operator = operator + i
                operator = operator + ", "
            for i in weights:
                operator = operator + i
                operator = operator + ", "
            for i in inputs_dim:
                operator = operator + i
                operator = operator + ", "
            for i in outputs_dim:
                operator = operator + i
                operator = operator + ", "
            for i in weights_dim:
                operator = operator + i
                operator = operator + ", "
            for key in attributes:
                operator = operator + attributes[key]
                operator = operator + ", "
        operator = operator.strip(', ')
        operator = operator + ");\n"

        fm.write(operator)
        fm.write(free_str)
    fm.write("}")
    fw.close()  
    fm.close() 


def main():
    paddle.set_device("cpu")
    FLAGS = parse_args()
    cfg = load_config(FLAGS.config)
    merge_config(FLAGS.opt)

    if FLAGS.slim_config:
        cfg = build_slim_model(cfg, FLAGS.slim_config, mode='test')

    # FIXME: Temporarily solve the priority problem of FLAGS.opt
    merge_config(FLAGS.opt)
    check_config(cfg)
    if 'use_gpu' not in cfg:
        cfg.use_gpu = False
    check_gpu(cfg.use_gpu)
    check_version()

    run(FLAGS, cfg)


if __name__ == '__main__':
    main()
