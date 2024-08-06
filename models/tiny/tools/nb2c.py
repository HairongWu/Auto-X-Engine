import os
from naivebuffer import *

op_mapping = {
    'conv2d':'autox_conv2d','depthwise_conv2d':'autox_conv2d','pool2d':'autox_pool2d',
    'concat':'autox_concat','transpose2':'autox_transpose',
    'split':'autox_split','matmul_v2':'autox_matmul',
    'elementwise_add':'autox_elementwise_add','softmax':'autox_softmax',
    'hard_sigmoid':'autox_hard_sigmoid', 'elementwise_mul':'autox_elementwise_mul',
    'nearest_interp_v2':'autox_nearest_interp', 'sigmoid':'autox_sigmoid',
    'scale':'autox_scale','sqrt':'autox_sqrt','elementwise_div':'autox_elementwise_div',
    'multiclass_nms3':'autox_multiclass_nms3','relu':'autox_relu',
    'fusion_elementwise_add_activation':'autox_fusion_elementwise_add_activation',
    'bilinear_interp_v2':'autox_bilinear_interp','arg_max':'autox_argmax',
    'swish':'autox_swish','layer_norm':'autox_layer_norm', 'slice':'autox_slice'
    , 'hard_swish':'autox_hard_swish', 'reduce_mean':'autox_reduce_mean', 'linear_interp_v2':'autox_linear_interp_v2'
    , 'relu6':'autox_relu6'
}

attrs = ['act_type','dilations','groups','paddings','strides','adaptive','ksize','pooling_type','axis','start_axis','stop_axis',
             'trans_x','trans_y','scale','align_corners','offset','slope','bias','bias_after_scale','beta','begin_norm_axis'
             ,'epsilon','axes','decrease_axis','ends','starts','threshold']

ignore_layers = ['feed','shape', 'fill_constant','reshape2','flatten_contiguous_range', 'fetch', 'assign', 'squeeze2']

def nb2c(output_dir, ops, weights_dict, dim_dict):
    fw = open(os.path.join(output_dir, "model.bin"), "wb")
    fm = open(os.path.join(output_dir, "model.h"), "w")
    index = 0
    para_index = 0
    fm.write('void model(const uint8_t *image, const float *weights, uint32_t *Out)\n{\n')
    operator = ''
    for key in dim_dict:
        if len(dim_dict[key]) > 0 and 'fill_constant' not in key:
            operator = operator + '\tuint16_t '
            operator = operator + key.replace('.','_')
            operator = operator + '_dim[] = {'+', '.join([str(abs(i)) for i in dim_dict[key]])+'};\n'
    fm.write(operator)
    fm.write('\n')

    for op in ops:
        if op.type in ignore_layers:
            continue

        operator = ''
        
        for i in op.inputs:
            if op.type == "softmax" or  op.type == "hard_sigmoid" or op.type == "swish":
                continue

            if 'XShape' == i.parameter or 'Bias' == i.parameter:
                continue

            if len(i.arguments) > 1:
                operator = operator + '\tfloat* p_%d[] = {'%(para_index)
                for arg in i.arguments:
                    operator = operator + arg.replace('.','_')
                    operator = operator + ', '
                operator = operator + '};\n'
                operator = operator + '\tuint16_t* p_%d_dim[] = {'%(para_index)
                for arg in i.arguments:
                    operator = operator + arg.replace('.','_')
                    operator = operator + '_dim'
                    operator = operator + ', '
                operator = operator + '};\n'

        for i in op.outputs:
            if op.type == "softmax" or  op.type == "hard_sigmoid" or op.type == "swish" \
                or op.type == "scale" or op.type == "sqrt" or op.type == "sigmoid" or op.type == "relu":
                continue
            if 'XShape' == i.parameter:
                continue
            for arg in i.arguments:
                operator = operator + '\tfloat *'
                operator = operator + arg.replace('.','_')
                operator = operator + " = (float *)calloc(%d, sizeof(float));\n"%abs(reduce(mul, dim_dict[arg]))
            if len(i.arguments) > 1:
                operator = operator + '\tfloat* p_%d[] = {'%(para_index)
                for arg in i.arguments:
                    operator = operator + arg.replace('.','_')
                    operator = operator + ', '
                operator = operator + '};\n'
                operator = operator + '\tuint16_t* p_%d_dim[] = {'%(para_index)
                for arg in i.arguments:
                    operator = operator + arg.replace('.','_')
                    operator = operator + '_dim'
                    operator = operator + ', '
                operator = operator + '};\n'

        
        inputs = []
        weights = []
        weights_name = []
        outputs = []

        attributes = {}

        for i in op.inputs:
            for arg in i.arguments:
                if arg in weights_dict:
                    for f in weights_dict[arg]:
                        fw.write(struct.pack('f',f))
                    weights.append("(float*)((int8_t*)weights) + " + str(index))
                    index = index + abs(reduce(mul, dim_dict[arg]))
                    weights_name.append(arg)
                else:
                    inputs.append(arg)
        for i in op.outputs:
             if 'XShape' == i.parameter:
                continue
             for arg in i.arguments:
                outputs.append(arg)

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
        
        if op.type == "transpose2":
            operator = operator + '\tuint16_t axis_%d'%(para_index)
            operator = operator + '[] = {'
            operator = operator + attributes['axis']
            operator = operator + '};\n'

        operator = operator +"\t"+ op_mapping[op.type]
        operator = operator + "("
        free_str = ''
        if op.type == "conv2d" or op.type == "depthwise_conv2d":
            for i in inputs:
                operator = operator + i.replace('.','_')
                operator = operator + ", "
                free_str = free_str + '\tfree(%s);\n'%(i.replace('.','_'))
            for i in weights:
                operator = operator + i.replace('.','_')
                operator = operator + ", "
            for i in outputs:
                operator = operator + i.replace('.','_')
                operator = operator + ", "
            
            for i in inputs:
                operator = operator + i.replace('.','_')
                operator = operator + "_dim, "

            operator = operator + weights_name[1].replace('.','_')
            operator = operator + "_dim, "
            for i in outputs:
                operator = operator + i.replace('.','_')
                operator = operator + "_dim, "

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
                elif '"relu6"' == attributes['act_type']:
                    operator = operator + str(ACTIVATION.kRelu6.value)
                else:
                    print(attributes['act_type'])
            else:
                operator = operator + str(ACTIVATION.kIndentity.value)
            operator = operator + ", "
        elif op.type == "pool2d":
            for i in inputs:
                operator = operator + i.replace('.','_')
                operator = operator + ", "
                free_str = free_str + '\tfree(%s);\n'%(i.replace('.','_'))
            for i in outputs:
                operator = operator + i.replace('.','_')
                operator = operator + ", "
            for i in inputs:
                operator = operator + i.replace('.','_')
                operator = operator + "_dim, "
            for i in outputs:
                operator = operator + i.replace('.','_')
                operator = operator + "_dim, "

            operator = operator + attributes['ksize'][0]
            operator = operator + ", "
            operator = operator + attributes['strides'][0]
            operator = operator + ", "
            operator = operator + attributes['paddings'][0]
            operator = operator + ", "
            operator = operator + attributes['adaptive'][0]
            operator = operator + ", "

            if attributes['pooling_type'] == '"avg"':
                operator = operator + str(1)
            else:
                operator = operator + str(0)
            operator = operator + ", "
        elif op.type == "transpose2":
            for i in inputs:
                operator = operator + i.replace('.','_')
                operator = operator + ", "
                free_str = free_str + '\tfree(%s);\n'%(i.replace('.','_'))
            for i in outputs:
                operator = operator + i.replace('.','_')
                operator = operator + ", "
            for i in inputs:
                operator = operator + i.replace('.','_')
                operator = operator + "_dim, "
            for i in outputs:
                operator = operator + i.replace('.','_')
                operator = operator + "_dim, "

            operator = operator + "axis_%d, "%(para_index)
            operator = operator + str(len(attributes['axis'].split(',')))
            operator = operator + ", "
        elif op.type == "hard_sigmoid" or op.type == "swish" \
            or op.type == "scale" or op.type == "sqrt" \
            or op.type == "sigmoid" or op.type == "relu"\
            or op.type == "hard_swish" or op.type == "relu6":
            for i in inputs:
                operator = operator + i.replace('.','_')
                operator = operator + ", "
            
            for i in inputs:
                operator = operator + i.replace('.','_')
                operator = operator + "_dim, "
            
            operator = operator + str(len(dim_dict[inputs[0]]))
            operator = operator + ", "
            
            for key in attributes:
                operator = operator + attributes[key]
                operator = operator + ", "
        elif op.type == "softmax":
            for i in inputs:
                operator = operator + i.replace('.','_')
                operator = operator + ", "
            
            for i in inputs:
                operator = operator + i.replace('.','_')
                operator = operator + "_dim[0], "
            for i in inputs:
                operator = operator + i.replace('.','_')
                operator = operator + "_dim[1], "
            

        elif op.type == "concat":
            operator = operator + 'p_%d'%(para_index)
            operator = operator + ", "
            for i in inputs:
                free_str = free_str + '\tfree(%s);\n'%(i.replace('.','_'))
            for i in outputs:
                operator = operator + i.replace('.','_')
                operator = operator + ", "
            
            operator = operator + "p_%d_dim, "%(para_index)

            for i in outputs:
                operator = operator + i.replace('.','_')
                operator = operator + "_dim, "
            for key in attributes:
                operator = operator + attributes[key]
                operator = operator + ", "
            
            operator = operator + str(len(inputs))
            operator = operator + ", "

            operator = operator + str(len(dim_dict[inputs[0]]))
            operator = operator + ", "
        elif op.type == "split":
            for i in inputs:
                operator = operator + i.replace('.','_')
                operator = operator + ", "
                free_str = free_str + '\tfree(%s);\n'%(i.replace('.','_'))
            
            operator = operator + 'p_%d'%(para_index)
            operator = operator + ", "

            for i in inputs:
                operator = operator + i.replace('.','_')
                operator = operator + "_dim, "
            
            operator = operator + "p_%d_dim, "%(para_index)

            for key in attributes:
                operator = operator + attributes[key]
                operator = operator + ", "
            operator = operator + str(len(dim_dict[inputs[0]]))
            operator = operator + ", "
            operator = operator + str(len(outputs))
            operator = operator + ", "

            operator = operator + str(len(dim_dict[outputs[0]]))
            operator = operator + ", "

        elif op.type == "elementwise_add" or op.type == "matmul_v2" or op.type == "elementwise_mul"\
            or op.type == "elementwise_div" or op.type == "fusion_elementwise_add_activation":
            for i in inputs:
                operator = operator + i.replace('.','_')
                operator = operator + ", "
                free_str = free_str + '\tfree(%s);\n'%(i.replace('.','_'))
            for i in weights:
                operator = operator + i.replace('.','_')
                operator = operator + ", "
            for i in outputs:
                operator = operator + i.replace('.','_')
                operator = operator + ", "
            
            for i in inputs:
                operator = operator + i.replace('.','_')
                operator = operator + "_dim, "
            for i in weights_name:
                operator = operator + i.replace('.','_')
                operator = operator + "_dim, "
            for i in outputs:
                operator = operator + i.replace('.','_')
                operator = operator + "_dim, "
            
            for key in attributes:
                operator = operator + attributes[key]
                operator = operator + ", "
            for i in inputs:
                operator = operator + str(len(dim_dict[i]))
                operator = operator + ", "
            if len(weights_name) > 0:
                operator = operator + str(len(dim_dict[weights_name[0]]))
                operator = operator + ", "
            operator = operator + str(len(dim_dict[outputs[0]]))
            operator = operator + ", "
        elif op.type == "nearest_interp_v2" or op.type == "bilinear_interp_v2":
            for i in inputs:
                operator = operator + i.replace('.','_')
                operator = operator + ", "
                free_str = free_str + '\tfree(%s);\n'%(i.replace('.','_'))
            for i in outputs:
                operator = operator + i.replace('.','_')
                operator = operator + ", "
            
            for i in inputs:
                operator = operator + i.replace('.','_')
                operator = operator + "_dim, "
            for i in outputs:
                operator = operator + i.replace('.','_')
                operator = operator + "_dim, "
            
            if len(attributes['scale']) > 0:
                operator = operator + attributes['scale'][0]
            else:
                operator = operator + attributes['scale']
            operator = operator + ", "
            operator = operator + attributes['align_corners']
            operator = operator + ", "
        elif op.type == "layer_norm":
            for i in inputs:
                operator = operator + i.replace('.','_')
                operator = operator + ", "
                free_str = free_str + '\tfree(%s);\n'%(i.replace('.','_'))
            for i in weights:
                operator = operator + i
                operator = operator + ", "

            operator = operator + outputs[0].replace('.','_')
            operator = operator + ", "
            
            for i in inputs:
                operator = operator + i.replace('.','_')
                operator = operator + "_dim, "
            for i in weights_name:
                operator = operator + i.replace('.','_')
                operator = operator + "_dim, "

            operator = operator + outputs[0].replace('.','_')
            operator = operator + "_dim, "
            
            for key in attributes:
                operator = operator + attributes[key]
                operator = operator + ", "
        else:
            for i in inputs:
                operator = operator + i.replace('.','_')
                operator = operator + ", "
                free_str = free_str + '\tfree(%s);\n'%(i.replace('.','_'))
            for i in weights:
                operator = operator + i.replace('.','_')
                operator = operator + ", "
            for i in outputs:
                operator = operator + i.replace('.','_')
                operator = operator + ", "
            
            for i in inputs:
                operator = operator + i.replace('.','_')
                operator = operator + "_dim, "
            for i in weights:
                operator = operator + i
                operator = operator + "_dim, "
            for i in outputs:
                operator = operator + i.replace('.','_')
                operator = operator + "_dim, "
            
            for key in attributes:
                operator = operator + attributes[key]
                operator = operator + ", "
        operator = operator.strip(', ')
        operator = operator + ");\n"

        fm.write(operator)
        fm.write(free_str)
        fm.write('\n')

        para_index = para_index + 1
    fm.write("}")
    fw.close()  
    fm.close() 
