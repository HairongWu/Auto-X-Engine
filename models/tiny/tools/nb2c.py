import os
from naivebuffer import *

op_mapping = {
    'conv2d':'autox_conv2d','depthwise_conv2d':'autox_conv2d','pool2d':'autox_pool2d',
    'concat':'autox_concat','transpose2':'autox_transpose2',
    'split':'autox_split','matmul_v2':'autox_matmul_v2',
    'elementwise_add':'autox_elementwise_add','softmax':'autox_softmax',
    'hard_sigmoid':'autox_hard_sigmoid', 'elementwise_mul':'autox_elementwise_mul',
    'nearest_interp_v2':'autox_nearest_interp_v2', 'sigmoid':'autox_sigmoid',
    'scale':'autox_scale','sqrt':'autox_sqrt','elementwise_div':'autox_elementwise_div',
    'multiclass_nms3':'autox_multiclass_nms3','relu':'autox_relu',
    'fusion_elementwise_add_activation':'autox_fusion_elementwise_add_activation',
    'bilinear_interp_v2':'autox_bilinear_interp_v2','arg_max':'autox_arg_max',
    'swish':'autox_swish','layer_norm':'autox_layer_norm',
}

attrs = ['act_type','dilations','groups','paddings','strides','adaptive','ksize','pooling_type','axis','start_axis','stop_axis',
             'trans_x','trans_y']

ignore_layers = ['feed','shape','slice','fill_constant','reshape2','flatten_contiguous_range', 'fetch', 'assign', 'squeeze2']

def nb2c(output_dir, ops, weights_dict, dim_dict):
    fw = open(os.path.join(output_dir, "model.bin"), "wb")
    fm = open(os.path.join(output_dir, "model.h"), "w")
    index = 0
    fm.write('void model(const uint8_t *image, const uint16_t ssize_h, const uint16_t ssize_w, const float *weights, uint32_t *Out)\n{\n')
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
            if op.type == "softmax" or  op.type == "hard_sigmoid" or op.type == "swish":
                continue
            if 'XShape' == i.parameter:
                continue
            for arg in i.arguments:
                operator = operator + '\tfloat *'
                operator = operator + arg.replace('.','_')
                operator = operator + " = (float *)calloc(%d, sizeof(float));\n"%abs(reduce(mul, dim_dict[arg]))
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
                if i.type == AttributeType.INT or i.type == AttributeType.LONG:
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
            operator = operator + attributes['paddings'][0]
            operator = operator + ", "
            operator = operator + attributes['strides'][0]
            operator = operator + ", "
            operator = operator + attributes['dilations'][0]
            operator = operator + ", "

            if 'act_type' in attributes:
                if '"relu"' == attributes['act_type']:
                    operator = operator + str(ACTIVATION.kRelu.value)
                elif '"hard_swish"' == attributes['act_type']:
                    operator = operator + str(ACTIVATION.kHardSwish.value)
                else:
                    print(attributes['act_type'])
            else:
                operator = operator + str(ACTIVATION.kIndentity.value)
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

            operator = operator + attributes['ksize'][0]
            operator = operator + ", "
            operator = operator + attributes['strides'][0]
            operator = operator + ", "
            operator = operator + attributes['paddings'][0]
            operator = operator + ", "
            operator = operator + attributes['adaptive'][0]
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
        elif op.type == "softmax" or op.type == "hard_sigmoid" or op.type == "swish":
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
