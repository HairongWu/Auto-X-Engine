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
    predictor.save_optimized_model(config.Global.save_inference_dir+"model")


    input_tensor = predictor.get_input(0)
    input_tensor.from_numpy(np.ones([1,3, 224, 224]).astype('float32'))
    # input_tensor2 = predictor.get_input(1)
    # input_tensor2.from_numpy(np.ones([1, 2]).astype('float32'))

    predictor.run()

    output_tensor = predictor.get_output(0)
    print(output_tensor.numpy())

    ops, dim_dict, weights_dict = read_nb(config.Global.save_inference_dir+"model.nb")

    op_mapping = {'conv2d':'autox_conv2d','depthwise_conv2d':'autox_conv2d','pool2d':'autox_pool2d',
                  'concat':'autox_concat','transpose2':'autox_transpose2','split':'autox_split','matmul_v2':'autox_matmul_v2',
                  'elementwise_add':'autox_elementwise_add','softmax':'autox_softmax'}
    attrs = ['act_type','dilations','groups','paddings','strides','adaptive','ksize','pooling_type','axis','start_axis','stop_axis',
             'trans_x','trans_y']
    ignore_layers = ['feed','shape','slice','fill_constant','reshape2','flatten_contiguous_range', 'fetch']
    
    fw = open(config.Global.save_inference_dir+"model.bin", "wb")
    fm = open(config.Global.save_inference_dir+"model.h", "w")
    index = 0
    fm.write('void ShuffleNetV2_x0_25(const uint8_t *image, const uint16_t ssize_h, const uint16_t ssize_w, const float *weights, uint32_t *Out)\n{\n')
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
