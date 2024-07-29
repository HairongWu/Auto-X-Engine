#!/usr/bin/env python3
import struct
import numpy as np
from enum import IntEnum

from functools import reduce
from operator import mul

class fbs_reader():
    def __init__(self, data, offset, identifier):
        self._buffer = data
        self._dataView = data.data

        self._length = data.size
        self._offset = offset
        self._identifier = identifier

    @staticmethod
    def open(data, offset):
        if offset is None:
            offset = 0

        if data.size >= (offset + 8):
            reader = fbs_reader(data, offset, None)
            root = reader.uint32(offset) + offset

            if root < reader._length:
                start = root - reader.int32(root)
                if start > 0 and (start + 4) < reader._length:
                    last = reader.int16(start) + start
                    max = reader.int16(start + 2)
                    if last < reader._length:
                        valid = True
                        for i in range(start + 4, last, 2):
                            offset = reader.int16(i)
                            if offset >= max:
                                valid = False
                                break
                        
                        if valid:
                            identifier = reader.identifier()
                            return fbs_reader(data, offset, identifier)

        return None

    def root(self):
        return self.int32(0) + 0

    def identifier(self):
        if self._identifier is None:
            buffer = self._buffer[self._offset + 4:self._offset + 8]
            self._identifier = str(buffer) if np.min(buffer) >= 32 and np.max(buffer) <= 128 else ''
        
        return self._identifier

    def bool(self, offset):
        return bool(self.int8(offset))

    def bool_(self, position, offset, defaultValue):
        offset = self.__offset(position, offset)
        return self.bool(position + offset) if offset else defaultValue

    def bools_(self, position, offset):
        offset = self.__offset(position, offset)
        if offset:
            length = self.__vector_len(position + offset)
            offset = self.__vector(position + offset)
            array = [None]*length
            for i in range(length):
                array[i] = True if self.uint8(offset + i + 4) else False
            
            return array
        
        return []

    def int8(self, offset):
        return self.uint8(offset) << 24 >> 24

    def uint8(self, offset):
        return self._buffer[offset]
    
    def int16(self, offset):
        return np.frombuffer(self._dataView, np.int16, offset=offset, count=1)[0]

    def  int32(self, offset):
        return np.frombuffer(self._dataView, np.int32, offset=offset, count=1)[0]

    def int32_(self, position, offset, defaultValue):
        offset = self.__offset(position, offset)
        return self.int32(position + offset) if offset else defaultValue

    def uint32(self, offset):
        return np.frombuffer(self._dataView, np.uint32, offset=offset, count=1)[0]

    def int64(self, offset):
        return np.frombuffer(self._dataView, np.int64, offset=offset, count=1)[0]
    
    def int64_(self, position, offset, defaultValue):
        offset = self.__offset(position, offset)
        return self.int64(position + offset) if offset else defaultValue

    def int64s_(self, position, offset):
        offset = self.__offset(position, offset)
        if offset:
            length = self.__vector_len(position + offset)
            offset = self.__vector(position + offset)
            array = [None]*length
            for i in range(length):
                array[i] = self.int64(offset + (i << 3))
            
            return array
        
        return []

    def float32(self, offset):
        return np.frombuffer(self._dataView, np.float32, offset=offset, count=1)[0]

    def float32_(self, position, offset, defaultValue):
        offset = self.__offset(position, offset)
        return self.float32(position + offset) if offset else defaultValue
    
    def float64(self, offset):
        return np.frombuffer(self._dataView, np.float64, offset=offset, count=1)[0]

    def float64_(self, position, offset, defaultValue):
        offset = self.__offset(position, offset)
        return self.float64(position + offset) if offset else defaultValue

    def reader_table(self, position, offset, decode):
        offset = self.__offset(position, offset)
        return decode(self, self.__indirect(position + offset)) if offset else None


    def string(self, offset, encoding):
        offset += self.int32(offset)
        length = self.int32(offset)
        result = ''
        i = 0
        offset += 4
        if encoding == 1:
            return self.data.subarray(offset, offset + length)
        
        while i < length:
            codePoint = ''
            # Decode UTF-8
            a = self.uint8(offset + i)
            i = i + 1
            if (a < 0xC0):
                codePoint = a
            else:
                b = self.uint8(offset + i)
                i = i + 1
                if (a < 0xE0):
                    codePoint = ((a & 0x1F) << 6) | (b & 0x3F)
                else:
                    c = self.uint8(offset + i)
                    i = i + 1
                    if (a < 0xF0):
                        codePoint = ((a & 0x0F) << 12) | ((b & 0x3F) << 6) | (c & 0x3F)
                    else:
                        d = self.uint8(offset + i)
                        i = i + 1
                        codePoint = ((a & 0x07) << 18) | ((b & 0x3F) << 12) | ((c & 0x3F) << 6) | (d & 0x3F)
                
                
            
            # Encode UTF-16
            if codePoint < 0x10000:
                result += chr(codePoint)
            else:
                codePoint -= 0x10000
                result += chr((codePoint >> 10) + 0xD800, (codePoint & ((1 << 10) - 1)) + 0xDC00)
            
        return result

    def string_(self, position, offset, defaultValue):
        offset = self.__offset(position, offset)
        return self.string(position + offset, 0) if offset else defaultValue
    
    def strings_(self, position, offset):
        offset = self.__offset(position, offset)
        if offset:
            length = self.__vector_len(position + offset)
            offset = self.__vector(position + offset)
            array = [None]*length
            for i in range(length):
                array[i] = self.string(offset + i * 4, 0)
            
            return array
        
        return []

    def __union(self, offset):
        return offset + self.int32(offset)

    def table(self, position, offset, decode):
        offset = self.__offset(position, offset)
        return decode(self, self.__indirect(position + offset)) if offset else None

    def union(self, position, offset, decode):
        type_offset = self.__offset(position, offset)
        type = self.uint8(position + type_offset) if type_offset else 0
        offset = self.__offset(position, offset + 2)
        return decode(self, self.__union(position + offset), type) if offset else None

    def typedArray(self, position, offset, type):
        offset = self.__offset(position, offset)
        return np.frombuffer(self._dataView, type, offset=self.__vector(position + offset), count= self.__vector_len(position + offset)) if offset else  0

    def tableArray(self, position, offset, decode):
        offset = self.__offset(position, offset)
        length = self.__vector_len(position + offset) if offset else 0
        list = [None]*length
        for i in range(length):
            list[i] = decode(self, self.__indirect(self.__vector(position + offset) + i * 4))
        
        return list

    def __vector(self, offset):
        return offset + self.int32(offset) + 4

    def __vector_len(self, offset):
        return self.int32(offset + self.int32(offset))
    
    def  __indirect(self, offset):
        return offset + self.int32(offset)

    def __offset(self, bb_pos, vtableOffset):
        vtable = bb_pos - self.int32(bb_pos)
        return self.int16(vtable + vtableOffset) if vtableOffset < self.int16(vtable) else 0

class VersionDesc():
    @staticmethod
    def decode(reader, position):
        version = reader.int32_(position, 4, 0)
        model_version = reader.int32_(position, 6, 0)

class ParamDesc_LoDTensorDesc():
    def __init__(self, lod, dim, data, data_type):
        self.lod = lod
        self.dim = dim
        self.data = data
        self.data_type = data_type

    @staticmethod
    def decode(reader, position):
        lod_level = reader.int32_(position, 4, 0)
        lod = reader.int64s_(position, 6)
        dim = reader.int64s_(position, 8)
        data_type = reader.int32_(position, 10, 0)
        data = reader.typedArray(position, 12, np.int8)

        return ParamDesc_LoDTensorDesc(lod, dim, data, data_type)

class VariableDesc():
    @staticmethod
    def decode(reader, position, type=1):
        if type == 1:
            return ParamDesc_LoDTensorDesc.decode(reader, position)
        else:
            return None

class TensorDesc():
    def __init__(self, dims):
        self.dims = dims

    @staticmethod
    def decode(reader, position):
        data_type = reader.int32_(position, 4, 0)
        dims = reader.int64s_(position, 6)

        return TensorDesc(dims)

class LoDTensorArrayDesc():
    def __init__(self, tensor):
        self.tensor = tensor

    @staticmethod
    def decode(reader, position):
        tensor = reader.table(position, 4, TensorDesc.decode)
        lod_level = reader.int32_(position, 6, 0)

        return LoDTensorArrayDesc(tensor)

class ReaderDesc():
    @staticmethod
    def decode(reader, position):
        lod_tensor = reader.tableArray(position, 4, VarType_LoDTensorDesc.decode)

class Tuple():
    @staticmethod
    def decode(reader, position):
        element_type = reader.typedArray(position, 4, np.int32)

class VarType_LoDTensorDesc():
    def __init__(self, tensor):
        self.tensor = tensor

    @staticmethod
    def decode(reader, position):
        tensor = reader.table(position, 4, TensorDesc.decode)
        lod_level = reader.int32_(position, 6, 0)

        return VarType_LoDTensorDesc(tensor)

class VarType():
    def __init__(self, type, lod_tensor):
        self.type = type
        self.lod_tensor = lod_tensor

    @staticmethod
    def decode(reader, position):
        type = reader.int32_(position, 4, 0)
        selected_rows = reader.table(position, 6, TensorDesc.decode)
        lod_tensor = reader.table(position, 8, VarType_LoDTensorDesc.decode)
        tensor_array = reader.table(position, 10, LoDTensorArrayDesc.decode)
        r = reader.table(position, 12, ReaderDesc.decode)
        tuple = reader.table(position, 14, Tuple.decode)

        return VarType(type, lod_tensor)

class VarDesc():
    def __init__(self, name, type, persistable):
        self.name = name
        self.type = type
        self.persistable = persistable

    @staticmethod
    def decode(reader, position):
        name = reader.string_(position, 4, None)
        type = reader.table(position, 6, VarType.decode)
        persistable = reader.bool_(position, 8, False)
        need_check_feed = reader.bool_(position, 10, False)

        return VarDesc(name, type, persistable)

class Var():
    def __init__(self, parameter, arguments):
        self.parameter = parameter
        self.arguments = arguments

    @staticmethod
    def decode(reader, position):
        parameter = reader.string_(position, 4, None)
        arguments = reader.strings_(position, 6)

        return Var(parameter, arguments)

class Attr():
    def __init__(self, name, type, i, f, s, ints, floats, b):
        self.name = name
        self.type = type
        self.i = i
        self.f = f
        self.s = s
        self.ints = ints
        self.floats = floats
        self.b = b

    @staticmethod
    def decode(reader, position):
        name = reader.string_(position, 4, None)
        type = reader.int32_(position, 6, 0)
        i = reader.int32_(position, 8, 0)
        f = reader.float32_(position, 10, 0)
        s = reader.string_(position, 12, None)
        ints = reader.typedArray(position, 14, np.int32)
        floats = reader.typedArray(position, 16, np.float32)
        strings = reader.strings_(position, 18)
        b = reader.bool_(position, 20, False)
        bools = reader.bools_(position, 22)
        block_idx = reader.int32_(position, 24, 0)
        l = reader.int64_(position, 26, 0)
        blocks_idx = reader.typedArray(position, 28, np.int32)
        longs = reader.int64s_(position, 30)
        float64 = reader.float64_(position, 32, 0)
        float64s = reader.typedArray(position, 34, np.float64)

        return Attr(name, type, i, f, s, ints, floats, b)
        
class OpDesc():
    def __init__(self, type, inputs, outputs, attrs):
        self.type = type
        self.inputs = inputs
        self.outputs = outputs
        self.attrs = attrs

    @staticmethod
    def decode(reader, position):
        type = reader.string_(position, 4, None)
        inputs = reader.tableArray(position, 6, Var.decode)
        outputs = reader.tableArray(position, 8, Var.decode)
        attrs = reader.tableArray(position, 10, Attr.decode)
        is_target = reader.bool_(position, 12, None)

        return OpDesc(type, inputs, outputs, attrs)

class BlockDesc():
    def __init__(self, vars, ops):
        self.vars = vars
        self.ops = ops

    @staticmethod
    def decode(reader, position):
        idx = reader.int32_(position, 4, 0)
        parent_idx = reader.int32_(position, 6, 0)
        vars = reader.tableArray(position, 8, VarDesc.decode)
        ops = reader.tableArray(position, 10, OpDesc.decode)
        forward_block_idx = reader.int32_(position, 12, -1)

        return BlockDesc(vars, ops)



class Version():
    @staticmethod
    def decode(reader, position):
        version = reader.int64_(position, 4, 0)

class OpVersion():
    @staticmethod
    def decode(reader, position):
        version = reader.int32_(position, 4, 0)

class OpVersionPair():
    @staticmethod
    def decode(reader, position):
        op_name = reader.string_(position, 4, None)
        op_version = reader.table(position, 6, OpVersion.decode)

class OpVersionMap():
    @staticmethod
    def decode(reader, position):
        pair = reader.tableArray(position, 4, OpVersionPair.decode)

class DataType(IntEnum):
    BOOL = 0,
    INT16 = 1,
    INT32 = 2,
    INT64 = 3,
    FP16 = 4,
    FP32 = 5,
    FP64 = 6,
    LOD_TENSOR = 7,
    SELECTED_ROWS = 8,
    FEED_MINIBATCH = 9,
    FETCH_LIST = 10,
    STEP_SCOPES = 11,
    LOD_RANK_TABLE = 12,
    LOD_TENSOR_ARRAY = 13,
    PLACE_LIST = 14,
    READER = 15,
    RAW = 17,
    TUPLE = 18,
    SIZE_T = 19,
    UINT8 = 20,
    INT8 = 21,
    BF16 = 22,
    COMPLEX64 = 23,
    COMPLEX128 = 24,

class AttributeType(IntEnum):
    INT = 0,
    FLOAT = 1,
    STRING = 2,
    INTS = 3,
    FLOATS = 4,
    STRINGS = 5,
    BOOLEAN = 6,
    BOOLEANS = 7,
    BLOCK = 8,
    LONG = 9,
    BLOCKS = 10,
    LONGS = 11,
    FLOAT64S = 12,
    VAR = 13,
    VARS = 14,
    FLOAT64 = 15

class ACTIVATION(IntEnum):
    kIndentity = 0,
    kRelu = 1,
    kRelu6 = 2,
    kPRelu = 3,
    kLeakyRelu = 4,
    kSigmoid = 5,
    kTanh = 6,
    kSwish = 7,
    kExp = 8,
    kAbs = 9,
    kHardSwish = 10,
    kReciprocal = 11,
    kThresholdedRelu = 12,
    kElu = 13,
    kHardSigmoid = 14,
    kLog = 15,
    kSigmoid_v2 = 16,
    kTanh_v2 = 17,
    kGelu = 18,
    kErf = 19,
    kSign = 20,
    kSoftPlus = 21,
    kMish = 22,
    kSilu = 23,
    kLog1p = 24,
    NUM = 25,

class POOLING_TYPE(IntEnum):
    MAX_POOLING = 0,
    AVG_POOLING = 1,

def read_nb(filename):

        f = open(filename, 'rb')

        meta_version = np.fromfile(f, dtype=np.uint16, count=1)[0] # meta_version
        #print(meta_version)
        opt_version = np.fromfile(f, dtype=np.int8, count=16) # opt_version
        #vprint(opt_version)

        topo_size = np.fromfile(f, dtype=np.uint64, count=1)[0]

        topo_table = np.fromfile(f, dtype=np.uint8, count=topo_size)
        reader = fbs_reader.open(topo_table,0)
        position = reader.root()

        blocks = reader.tableArray(position, 4, BlockDesc.decode)

        version = reader.table(position, 6, Version.decode)
        op_version_map = reader.table(position, 8, OpVersionMap.decode)
        
        f.read(4) # skip header
        header_size = np.fromfile(f, dtype=np.uint16, count=1)[0]
        params_size = np.fromfile(f, dtype=np.uint16, count=1)[0]

        max_tensor_size = np.fromfile(f, dtype=np.uint32, count=1)[0] # max_tensor_size
        f.read(header_size-6) # skip header

        weights_dict = {}
        for i in range(params_size):

            total_size = np.fromfile(f, dtype=np.uint32, count=1)[0]
            offset = np.fromfile(f, dtype=np.uint32, count=1)[0]

            param_bytes = total_size - offset
            param_data = np.fromfile(f, dtype=np.uint8, count=param_bytes)

            reader = fbs_reader.open(param_data,0)

            position = reader.root()
            version = reader.table(position, 4, VersionDesc.decode)
            name = reader.string_(position, 6, None)
            variable = reader.union(position, 8, VariableDesc.decode);   

            if variable.data_type == DataType.FP32:
                weights_dict[name] = np.frombuffer(variable.data.data, np.float32, offset=0, count=-1)  
            elif variable.data_type == DataType.INT8:
                weights_dict[name] = variable.data

        ops = []
        dim_dict = {}
        for block in blocks:
            
            for var in block.vars:
                dim_dict[var.name] = var.type.lod_tensor.tensor.dims

            for op in block.ops:
                ops.append(op)

        f.close()

        return ops, dim_dict, weights_dict
