from random import random

import torch
import pandas as pd
import ast
import numpy as np
import json
import onnx
import argparse

from PIL import Image
from multipart import file_path

import conv_divide_class as conv1
import pool_divide_class as pool1
import fc_divide_class as fc1
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

class OrtRunInfoExtractor(ast.NodeVisitor):
    def __init__(self):
        self.run_inputs = {}  # 用于存储ort_session.run函数调用的输入信息，键为变量名，值为具体值
        self.img_path=None
    def visit_Call(self, node):
        if isinstance(node.func, ast.Attribute) and node.func.attr == 'run':
            if node.args and isinstance(node.args[1], ast.Dict):
                for key, value in zip(node.args[1].keys, node.args[1].values):
                    if isinstance(key, ast.Constant) and isinstance(value, ast.Name):
                        var_name = value.id
                        self.find_variable_value(var_name)

    def find_variable_value(self, var_name):
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Assign):
                if isinstance(node.targets[0], ast.Name) and node.targets[0].id == var_name:
                    value = node.value
                    if isinstance(value, ast.AST):
                        value = self.resolve_value(value)
                    #这里只捕获第一个来源。。。
                    if var_name not in self.run_inputs.keys():
                        self.run_inputs[var_name] = value

 
    def resolve_value(self, node):
            if isinstance(node, ast.Constant):
                return node.value
            elif isinstance(node, ast.List):
                return [self.resolve_value(elt) for elt in node.elts]
            elif isinstance(node, ast.Tuple):
                return tuple(self.resolve_value(elt) for elt in node.elts)
            elif isinstance(node, ast.Dict):
                return {self.resolve_value(key): self.resolve_value(value) for key, value in zip(node.keys, node.values)}
            elif isinstance(node, ast.Name):
                return self.variables.get(node.id, None)
            elif isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Attribute):
                    if isinstance(func.value, ast.Call):
                        inner_result = self.resolve_value(func.value)
                        if isinstance(inner_result, np.ndarray):
                            if func.attr == 'astype':
                                dtype = self.resolve_value(node.args[0])
                                return inner_result.astype(dtype)
                    elif isinstance(func.value, ast.Attribute):
                        if isinstance(func.value.value, ast.Name) and func.value.value.id == 'np':
                            if func.value.attr == 'random' and func.attr in np.random.__all__:
                                args_resolved = [self.resolve_value(arg) for arg in node.args]
                                return generate_random_input(func.attr, args_resolved)
                            elif func.value.attr == 'resize':
                                data = self.resolve_value(node.args[0])
                                new_shape = self.resolve_value(node.args[1])
                                try:
                                    original_data = np.array(data)
                                    return np.resize(original_data, new_shape)
                                except ValueError as e:
                                    print(f"resize操作错误：{e}")
                                    return None
                            elif func.value.attr == 'reshape':
                                data = self.resolve_value(node.args[0])
                                new_shape = self.resolve_value(node.args[1])
                                try:
                                    original_data = np.array(data)
                                    return original_data.reshape(new_shape)
                                except ValueError as e:
                                    print(f"reshape操作错误：{e}")
                                return None
                            elif func.value.attr == 'transpose':
                                data = self.resolve_value(node.args[0])
                                axes = [self.resolve_value(elt) for elt in node.args[1].elts]
                                try:
                                    original_data = np.array(data)
                                    if len(original_data.shape)!= len(axes):
                                        raise ValueError(
                                            f"输入数据维度与轴参数数量不匹配。数据维度：{original_data.shape}，轴参数：{axes}")
                                    return original_data.transpose(axes)
                                except ValueError as e:
                                    print(f"transpose操作错误：{e}")
                                    return None
                            elif func.value.attr == 'squeeze':
                                data = self.resolve_value(node.args[0])
                                axis = self.resolve_value(node.args[1]) if len(node.args) > 0 else None
                                try:
                                    original_data = np.array(data)
                                    return original_data.squeeze(axis)
                                except ValueError as e:
                                    print(f"squeeze操作错误：{e}")
                                    return None
                            elif func.value.attr == 'expand_dims':
                                data = self.resolve_value(node.args[0])
                                axis = self.resolve_value(node.args[1]) if len(node.args) > 0 else None
                                try:
                                    original_data = np.array(data)
                                    return original_data.expand_dims(axis)
                                except ValueError as e:
                                    print(f"expand_dims操作错误：{e}")
                                    return None
                            elif func.value.attr == 'flatten':
                                data = self.resolve_value(node.args[0])
                                return np.flatten(data)
                            elif func.value.attr == 'repeat':
                                data = self.resolve_value(node.args[0])
                                repeats = self.resolve_value(node.args[1])
                                axis = self.resolve_value(node.args[2]) if len(node.args) > 2 else None
                                try:
                                    original_data = np.array(data)
                                    return original_data.repeat(repeats, axis)
                                except ValueError as e:
                                    print(f"repeat操作错误：{e}")
                                return None
                            elif func.value.attr == 'tile':
                                data = self.resolve_value(node.args[0])
                                reps = self.resolve_value(node.args[1])
                                try:
                                    original_data = np.array(data)
                                    return np.tile(original_data, reps)
                                except ValueError as e:
                                    print(f"tile操作错误：{e}")
                                    return None
                        elif isinstance(func.value, ast.Name) and func.value.id == 'pd':
                            if func.attr == 'read_csv':
                                args_resolved = [self.resolve_value(arg) for arg in node.args]
                                return pd.read_csv(*args_resolved)
                            elif func.attr == 'apply':
                                df = self.variables.get(func.value.id, None)
                                func_to_apply = self.resolve_value(node.args[0])
                                args = [self.resolve_value(arg) for arg in node.args[1:]]
                                return df.apply(func_to_apply, axis=args[0], args=args[1:])



                    elif isinstance(func.value, ast.Name):
                        if func.value.id == 'np':
                            if func.attr == 'rand':
                                return np.random.rand(*self.resolve_value(node.args))
                            elif func.attr == 'randint':
                                return np.random.randint(*self.resolve_value(node.args))
                            elif func.attr == 'normal':
                                return np.random.normal(*self.resolve_value(node.args))
                            elif func.attr == 'randn':
                                return np.random.randn(*self.resolve_value(node.args))
                            elif func.attr == 'uniform':
                                return np.random.uniform(*self.resolve_value(node.args))
                        elif func.value.id == 'torch':
                            if func.attr == 'tensor':
                                data = self.resolve_value(node.args[0])
                                dtype = self.resolve_value(node.args[1]) if len(node.args) > 1 else None
                                device = self.resolve_value(node.args[2]) if len(node.args) > 2 else None
                                try:
                                    return torch.tensor(data, dtype=dtype, device=device)
                                except ValueError as e:
                                    print(f"tensor操作错误：{e}")
                                    return None
                            elif func.attr == 'from_numpy':
                                data = self.resolve_value(node.args[0])
                                try:
                                    return torch.from_numpy(data)
                                except ValueError as e:
                                    print(f"from_numpy操作错误：{e}")
                                    return None
                            elif func.attr == 'reshape':
                                data = self.resolve_value(node.args[0])
                                new_shape = self.resolve_value(node.args[1])
                                try:
                                    tensor = self.resolve_value(node.args[0])
                                    return tensor.reshape(new_shape)
                                except ValueError as e:
                                    print(f"reshape操作错误：{e}")
                                    return None
                            elif func.attr == 'transpose':
                                data = self.resolve_value(node.args[0])
                                axes = self.resolve_value(node.args[1])
                                try:
                                    tensor = self.resolve_value(node.args[0])
                                    return tensor.transpose(axes)
                                except ValueError as e:
                                    print(f"transpose操作错误：{e}")
                                    return None
                            elif func.attr == 'squeeze':
                                data = self.resolve_value(node.args[0])
                                axis = self.resolve_value(node.args[1]) if len(node.args) > 0 else None
                                try:
                                    tensor = self.resolve_value(node.args[0])
                                    return tensor.squeeze(axis)
                                except ValueError as e:
                                    print(f"squeeze操作错误：{e}")
                                    return None
                            elif func.attr == 'expand_dims':
                                data = self.resolve_value(node.args[0])
                                axis = self.resolve_value(node.args[1]) if len(node.args) > 0 else None
                                try:
                                    tensor = self.resolve_value(node.args[0])
                                    return tensor.expand_dims(axis)
                                except ValueError as e:
                                    print(f"expand_dims操作错误：{e}")
                                    return None
                            elif func.attr == 'flatten':
                                data = self.resolve_value(node.args[0])
                                try:
                                    tensor = self.resolve_value(node.args[0])
                                    return tensor.flatten()
                                except ValueError as e:
                                    print(f"flatten操作错误：{e}")
                                    return None
                            elif func.attr == 'repeat':
                                data = self.resolve_value(node.args[0])
                                repeats = self.resolve_value(node.args[1])
                                axis = self.resolve_value(node.args[2]) if len(node.args) > 2 else None
                                try:
                                    tensor = self.resolve_value(node.args[0])
                                    return tensor.repeat(repeats, axis)
                                except ValueError as e:
                                    print(f"repeat操作错误：{e}")
                                return None
                            elif func.attr == 'tile':
                                data = self.resolve_value(node.args[0])
                                reps = self.resolve_value(node.args[1])
                                try:
                                    tensor = self.resolve_value(node.args[0])
                                    return tensor.tile(reps)
                                except ValueError as e:
                                    print(f"tile操作错误：{e}")
                                return None
                            elif func.attr == 'cat':
                                tensors = [self.resolve_value(arg) for arg in node.args]
                                dim = self.resolve_value(node.args[1]) if len(node.args) > 1 else None
                                try:
                                    return torch.cat(tensors, dim)
                                except ValueError as e:
                                    print(f"cat操作错误：{e}")
                                    return None
                            elif func.attr == 'stack':
                                tensors = [self.resolve_value(arg) for arg in node.args]
                                dim = self.resolve_value(node.args[1]) if len(node.args) > 1 else None
                                try:
                                    return torch.stack(tensors, dim)
                                except ValueError as e:
                                    print(f"stack操作错误：{e}")
                                return None
                        elif func.value.id=="Image" :
                            if func.attr=="open":
                                args_resolved = [self.resolve_value(arg) for arg in node.args]
                                kwargs_resolved = {kw.arg: self.resolve_value(kw.value) for kw in node.keywords}
                                try:
                                    img = Image.open(*args_resolved, **kwargs_resolved)
                                    self.img_path=args_resolved[0]

                                    img_resized = img.resize((32, 32))
                                    if len(img_resized.size)==2:
                                        img_resized = img_resized.convert('RGB')
                                    img_array=np.array(img_resized)
                                    #将imgresize为(1,3,32,32)大小，首先img大小是（512,512,3）
                                    img_transposed=np.transpose(img_array,(2,0,1))
                                    img_final=np.expand_dims(img_transposed,axis=0)
                                    print("调整后图像数组大小：",img_final.shape)
                                    return img_final
                                except Exception as e:
                                    print(f"Image.open操作错误：{e}")
                                    return None
                        # elif func.attr=="resize":
                        #     args_resolved = [self.resolve_value(arg) for arg in node.args]

            else:
                return None
def generate_random_input(func_name, args):
    if hasattr(np.random, func_name):
        return getattr(np.random, func_name)(*args)
    return None
def reshape_and_pad_row(row):
# 将每行展平为一维
    row_flat = row.flatten()
    # 计算需要的行数
    total_elements = row_flat.size
    new_rows = (total_elements + 7) // 8  # 向上取整
    # 填充最后一行不足 8 个数的部分
    padded_row = np.pad(row_flat, (0, new_rows * 8 - total_elements), mode='constant', constant_values=0)
    
    # 重塑数组为新的形状
    reshaped_row = padded_row.reshape(new_rows, 8)
    
    return reshaped_row

def reshape_and_pad(data):
    reshaped_data = []
    for row in data:
        reshaped_row = reshape_and_pad_row(row)
        reshaped_data.append(reshaped_row)
    return np.vstack(reshaped_data)
class FrontEnd:
    def __init__(self, file_path, save_path,produce_file,option):

        self.ONNX_op_2_PIMCOMP_op = {
                "Conv": "OP_CONV",
                "Relu": "OP_RELU",
                "Tanh": "OP_TANH",
                "Sigmoid": "OP_SIGMOID",
                "MaxPool": "OP_POOL",
                "Flatten": "OP_FLATTEN",
                "Gemm": "OP_FC",
                "Dropout": "OP_DROPOUT",
                "LRN": "OP_LRN",
                "Concat": "OP_CONCAT",
                "AveragePool": "OP_POOL",
                "GlobalAveragePool": "OP_POOL",
                "Reshape": "OP_RESHAPE",
                "Transpose": "OP_TRANSPOSE",
                "Softmax": "OP_SOFTMAX",
                "BatchNormalization": "OP_BN",
                "Sum": "OP_ELTWISE",
                "Add": "OP_ELTWISE",
                "Sub": "OP_ELTWISE",
                "Mul": "OP_ELTWISE",
                "Pad": "OP_PAD",
                "Clip": "OP_CLIP",
                "Squeeze": "OP_SQUEEZE",
                "MatMul": "OP_MATMUL",
                "Shape": "OP_SHAPE",
                "Gather": "OP_GATHER",
                "Unsqueeze": "OP_UNSQUEEZE"}
        self.model_path = None
        self.save_path = save_path
        self.file_path=file_path
        self.model_name = None
        self.begin =0
        self.end=0
        self.memory_size = 10000000  
        self.variables1 = {}
        self.memory = torch.zeros(self.memory_size,8)
        self.produce_file = produce_file
        self.option=option
        self.img_path = None

    def extract_ort_run_info(self):
        with open(self.file_path, 'r', encoding='utf-8') as f:
            script_content = f.read()
        model_path='model'
        tree = ast.parse(script_content)
        for node in ast.walk(tree):
            # 记录所有变量的赋值情况
            if isinstance(node, ast.Assign):
                if isinstance(node.targets[0], ast.Name):
                    # var_name = node.targets[0].id
                    # value = node.value
                    # if isinstance(value, ast.AST):
                    #     value = resolve_ast_value(value, variables)
                    # variables[var_name] = value

                    # 查找模型路径的赋值
                    if (isinstance(node.value, ast.Constant) and
                        isinstance(node.value.value, str) and
                        node.value.value.endswith('.onnx')):
                        model_path = node.value.value

        extractor = OrtRunInfoExtractor()

        extractor.tree = tree
        extractor.visit(tree)
        self.img_path = extractor.img_path
        self.model_path=model_path
        self.model_name=self.model_path.split("/")[-1].split(".")[0]
        data=extractor.run_inputs
        data = np.array(list(data.values()))
        data = np.squeeze(data, axis=0)
        print(data.shape)
        self.input=data
        data = data.reshape(data.shape[0], data.shape[2]*data.shape[1], data.shape[3])
        data = data.reshape(data.shape[0]*data.shape[1],  data.shape[2])
        print(data.shape)
        data = reshape_and_pad(data)
        print(data.shape)
        self.begin = self.end  # Set the begin index
        if(len(data.shape)==1):
                self.end = self.begin + 1
                data=data.flatten()
        else:
            self.end = self.begin + data.shape[0]
        print(data.shape) # Calculate the end index
        print(self.begin,self.end) # Calculate the end index
        print(self.memory[self.begin:self.end].shape)
        self.memory[self.begin:self.end] = torch.from_numpy(data)  # Write the data to memo
        return model_path,extractor.run_inputs
    def load_model(self):
        self.model = onnx.load_model(self.model_path)
        #将输入[N,3,224,224]变为[1,3,224,224]
        # for input in self.model.graph.input:
        #     print(input.name)
        self.model.graph.input[0].type.tensor_type.shape.dim[0].dim_value = 1
        # self.model.graph.input[0].type.tensor_type.shape.dim[1].dim_value = 3
        # self.model.graph.input[0].type.tensor_type.shape.dim[2].dim_value = 224
        # self.model.graph.input[0].type.tensor_type.shape.dim[3].dim_value = 224
        self.model = onnx.shape_inference.infer_shapes(self.model)
        onnx.save(self.model, self.model_path)
        self.node_num = len(self.model.graph.node)
        # print(self.model.graph.value_info)
        for node in self.model.graph.node:
            if node.op_type == "Constant":
                print(node.name)

    def parse_model(self):
        # 对节点进行重命名
        for i in range(self.node_num):
            node = self.model.graph.node[i]
            self.model.graph.node[i].name = node.output[0]
        # 根据tensor的name获取其dim和dim_num
        self.get_dim_num_from_tensor_name = {}
        self.get_dim_from_tensor_name = {}
        for ioput in self.model.graph.value_info:
            name = ioput.name
            dim_num = len(ioput.type.tensor_type.shape.dim)
            dim = [ioput.type.tensor_type.shape.dim[i].dim_value for i in range(dim_num)]
            print("name=%r dim_num=%r dim=%r" % (name, dim_num, dim))
            self.get_dim_num_from_tensor_name[name] = dim_num
            self.get_dim_from_tensor_name[name] = dim
        # Pad节点预处理：得到constant节点对应的参数值
        self.pad_constant_node = {}
        for node in self.model.graph.node:
            if node.op_type == "Constant" and node.attribute[0].t.dims == [8]:
                self.pad_constant_node[node.name] = np.frombuffer(node.attribute[0].t.raw_data, dtype = np.int64)
        # 获取所有initializer_tensor的名称
        self.initializer_tensor = [tensor.name for tensor in self.model.graph.initializer]
        # 根据initializer_tensor的名称获取其index
        self.initializer_name_to_index = dict(zip(self.initializer_tensor,[i for i in range(len(self.initializer_tensor))]))
        self.constant_node = [node.name for node in self.model.graph.node if node.op_type == "Constant"]
        # 获取每个节点的生产者消费者信息（最终的output没有消费者，最初的data没有生产者）
        self.node_provider = {}
        self.node_consumer = {}
        for node in self.model.graph.node:
            for input_tensor in node.input:
                if not input_tensor in self.initializer_tensor and not input_tensor in self.pad_constant_node and not input_tensor in self.constant_node:
                    if node.name in self.node_provider:
                        self.node_provider[node.name] += [input_tensor]
                        if (self.node_consumer.get(input_tensor)):
                            self.node_consumer[input_tensor] += [node.name]
                        else:
                            self.node_consumer[input_tensor] = [node.name]
                    else:
                        self.node_provider[node.name] = [input_tensor]
                        if (self.node_consumer.get(input_tensor)):
                            self.node_consumer[input_tensor] += [node.name]
                        else:
                            self.node_consumer[input_tensor] = [node.name]
        # 根据tensor的name获取其dim和dim_num
        self.get_dim_num_from_tensor_name = {}
        self.get_dim_from_tensor_name = {}
        for tensor in self.model.graph.value_info:
            name = tensor.name
            dim_num = len(tensor.type.tensor_type.shape.dim)
            dim = [tensor.type.tensor_type.shape.dim[i].dim_value for i in range(dim_num)]
            self.get_dim_num_from_tensor_name[name] = dim_num
            self.get_dim_from_tensor_name[name] = dim

        # 增加input对应的tensor的dim_num和dim
        self.input_num = len(self.model.graph.input)
        self.input_names = [x.name for x in self.model.graph.input]
        for i in range(self.input_num):
            input_i = self.model.graph.input[i]
            name = input_i.name
            dim_num = len(input_i.type.tensor_type.shape.dim)
            dim = [input_i.type.tensor_type.shape.dim[i].dim_value for i in range(dim_num)]
            self.get_dim_num_from_tensor_name[name] = dim_num
            self.get_dim_from_tensor_name[name] = dim

        # 增加output对应的tensor的dim_num和dim
        self.output_num = len(self.model.graph.output)
        self.output_names = [x.name for x in self.model.graph.output]
        for i in range(self.output_num):
            output_i = self.model.graph.output[i]
            name = output_i.name
            dim_num = len(output_i.type.tensor_type.shape.dim)
            dim = [output_i.type.tensor_type.shape.dim[i].dim_value for i in range(dim_num)]
            self.get_dim_num_from_tensor_name[name] = dim_num
            self.get_dim_from_tensor_name[name] = dim

    def produce_info(self):
        self.node_list = []
        
        for i in range(self.input_num):
            input_node = self.model.graph.input[i]
            input_name = input_node.name
            if (input_name in self.initializer_tensor):
                continue
            input_node_info = {"bitwidth": 16,
                               "index": len(self.node_list),
                               "name": input_name,
                               "operation": "OP_INPUT",
                               "provider_num": 0,
                               "consumer_num": len(self.node_consumer[input_name]),
                               "consumer": self.node_consumer[input_name],
                                "begin" : self.begin,
                                "end" : self.end,   
                               "output_dim_num": self.get_dim_num_from_tensor_name[input_name],
                               "output_dim": self.get_dim_from_tensor_name[input_name],
                                "input_value":"self.input_value"
                               }
            input_node_info = dict(sorted(input_node_info.items()))
            self.node_list.append(input_node_info)
        i=0
        # 首先将每个节点命名为其output tensor的名字
        effective_input_num = len(self.node_list)
        for i in range(self.node_num):
            node = self.model.graph.node[i]

            # output_name (node_name)
            output_name = node.output[0]

            # operation
            operation = self.ONNX_op_2_PIMCOMP_op.get(node.op_type)
            if node.op_type == "Constant":
                continue
            elif operation == None:
                print("operation: ", node.op_type, " not considered")
                continue

            # output_dim_num
            output_dim_num = self.get_dim_num_from_tensor_name[output_name]
            output_dim = self.get_dim_from_tensor_name[output_name]

            node_info = {"bitwidth": 16,
                         "index": len(self.node_list),
                         "name": output_name,
                         "operation": operation,
                         "output_dim_num": output_dim_num,
                         "output_dim": output_dim
                         }

            # params
            params = {}
            attribute = node.attribute
            
            if node.op_type == "Conv":
                weight_tensor_name = node.input[1]
                
                for init in self.model.graph.initializer:
                    if init.name == weight_tensor_name:
                        variable_name = f"var_{i}"
                        self.variables1[variable_name] = np.frombuffer(init.raw_data, dtype=np.float32).reshape(init.dims)
                        self.begin = self.end  # Set the begin index
                        # print(f'var_{i}', self.variables1[variable_name].shape)
                        data = self.variables1[variable_name]
                        data = data.reshape(data.shape[0], data.shape[2]*data.shape[1], data.shape[3])
                        data = data.reshape(data.shape[0]*data.shape[1],  data.shape[2])
                        print(data.shape)
                        data = np.pad(data, ((0, 0), (0, 8-data.shape[1])), mode='constant', constant_values=0)
                        self.begin = self.end  # Set the beginning index
                        if(len(data.shape)==1):
                                self.end = self.begin + 1
                                data=data.flatten()
                        else:
                            self.end = self.begin + data.shape[0]
                        print(data.shape) # Calculate the end index
                        print(self.begin,self.end) # Calculate the end index
                        print(self.memory[self.begin:self.end].shape)
                        print(node.op_type)
                        self.memory[self.begin:self.end] = torch.from_numpy(data)  # Write the data to memory
                        
                        
                        
                
                # node_info["with_bn"] = 0
                # node_info["with_act"] = 0
                # node_info["act_type"] = -1
                # node_info["with_clip"] = 0
                # node_info["clip_min"] = -10000000
                # node_info["clip_max"] = 10000000
                for one_param in attribute:
                    if (one_param.name == "strides"):
                        params["stride_h"] = one_param.ints[0]
                        params["stride_w"] = one_param.ints[1]
                    if (one_param.name == "group"):
                        params["group"] = one_param.i
                    if (one_param.name == "pads"):
                        params["pad_h0"] = one_param.ints[0]
                        params["pad_h1"] = one_param.ints[2]
                        params["pad_w0"] = one_param.ints[1]
                        params["pad_w1"] = one_param.ints[3]
                    if (one_param.name == "kernel_shape"):
                        params["kernel_h"] = one_param.ints[0]
                        params["kernel_w"] = one_param.ints[1]
                    if (one_param.name == "dilations"):
                        params["dilation_h"] = one_param.ints[0]
                        params["dilation_w"] = one_param.ints[1]
                params["input_channel"] = self.get_dim_from_tensor_name[self.node_provider[output_name][0]][1]
                params["output_channel"] = output_dim[1]
                # params["weight_name"] = "var_"+str(i)
                params["weight_name"] = variable_name
                params["weight_name_begin"] = self.begin
                params["weight_name_end"] = self.end
                i+=1
                if len(node.input) == 2:
                    params["with_bias"] = 0
                elif len(node.input) == 3:
                    bias_tensor_name = node.input[2]
                    params["with_bias"] = 1
                    for init in self.model.graph.initializer:
                        if init.name == bias_tensor_name:
                            variable_name = f"var_{i}"
                            self.variables1[variable_name]= np.frombuffer(init.raw_data, dtype=np.float32).reshape(init.dims)
                            data = self.variables1[variable_name]
                            print(data.shape,init.dims)
                            if(len(data.shape)==1):
                                if(data.shape[0]<=8):
                                    data = np.pad(data, ((0, 8-data.shape[0])), mode='constant', constant_values=0)
                                else:
                                # 填充到 (3, 8)
                                    data = reshape_and_pad(data)
                            elif(len(data.shape)==2):
                                if(data.shape[1]<=8):
                                    data = np.pad(data, ((0, 0), (0, 8-data.shape[1])), mode='constant', constant_values=0)
                                else:
                                # 填充到 (3, 8)
                                    data = reshape_and_pad(data)
                           
                            elif(len(data.shape)==3):
                                data = data.reshape(data.shape[0]*data.shape[1],  data.shape[2])
                                if(data.shape[1]<=8):
                                    data = np.pad(data, ((0, 0), (0, 8-data.shape[1])), mode='constant', constant_values=0)
                                else:
                                # 填充到 (3, 8)
                                    data = reshape_and_pad(data)
                            elif (len(data.shape)==4):
                                data = data.reshape(data.shape[0], data.shape[2]*data.shape[1], data.shape[3])
                                data = data.reshape(data.shape[0]*data.shape[1],  data.shape[2])
                                print(data.shape)
                                if(data.shape[1]<=8):
                                    data = np.pad(data, ((0, 0), (0, 8-data.shape[1])), mode='constant', constant_values=0)
                                else:
                                # 填充到 (3, 8)
                                    data = reshape_and_pad(data)
                            else :
                                data = data.reshape(data.shape[0], -1, data.shape[-1])
                                data = data.reshape(data.shape[0] * data.shape[1], data.shape[2])
                                if(data.shape[1]<=8):
                                    data = np.pad(data, ((0, 0), (0, 8-data.shape[1])), mode='constant', constant_values=0)

                                else:
                                    data = reshape_and_pad(data)
                            self.begin = self.end  # Set the beginning index
                            if(len(data.shape)==1):
                                self.end = self.begin + 1
                                data=data.flatten()
                            else:
                                self.end = self.begin + data.shape[0]
                            print(data.shape) # Calculate the end index
                            print(self.begin,self.end) # Calculate the end index
                            print(self.memory[self.begin:self.end].shape)
                            self.memory[self.begin:self.end] = torch.from_numpy(data)  # Write the data to memory
                            
                             # Calculate the end index
                            print(f'var_{i}', self.variables1[variable_name].shape)
                            # params["with_bias_name"] = "var_" + str(i)
                            params["with_bias_name"] = variable_name
                            params["with_bias_begin"] = self.begin
                            params["with_bias_end"] = self.end

                            i += 1
            elif node.op_type == "MaxPool":
                for one_param in attribute:
                    if (one_param.name == "strides"):
                        params["stride_h"] = one_param.ints[0]
                        params["stride_w"] = one_param.ints[1]
                    if (one_param.name == "pads"):
                        params["pad_h0"] = one_param.ints[0]
                        params["pad_h1"] = one_param.ints[2]
                        params["pad_w0"] = one_param.ints[1]
                        params["pad_w1"] = one_param.ints[3]
                    if (one_param.name == "kernel_shape"):
                        params["kernel_h"] = one_param.ints[0]
                        params["kernel_w"] = one_param.ints[1]
                params["pool_method"] = 0
                params["global"] = 0
            elif node.op_type == "AveragePool":
                param_num = len(attribute)
                for one_param in attribute:
                    if (one_param.name == "strides"):
                        params["stride_h"] = one_param.ints[0]
                        params["stride_w"] = one_param.ints[1]
                    if (one_param.name == "pads"):
                        params["pad_h0"] = one_param.ints[0]
                        params["pad_h1"] = one_param.ints[2]
                        params["pad_w0"] = one_param.ints[1]
                        params["pad_w1"] = one_param.ints[3]
                    if (one_param.name == "kernel_shape"):
                        params["kernel_h"] = one_param.ints[0]
                        params["kernel_w"] = one_param.ints[1]
                params["pool_method"] = 1
                params["global"] = 0
            elif node.op_type == "GlobalAveragePool":
                params["stride_h"] = 1
                params["stride_w"] = 1
                params["pad_h0"] = 0
                params["pad_h1"] = 0
                params["pad_w0"] = 0
                params["pad_w1"] = 0
                params["kernel_h"] = self.get_dim_from_tensor_name[self.node_provider[output_name][0]][2]
                params["kernel_w"] = self.get_dim_from_tensor_name[self.node_provider[output_name][0]][3]
                params["pool_method"] = 1
                params["global"] = 1
            elif node.op_type == "Pad":
                # Pad在ONNX中可能有两种表示。所以分情况处理。
                if len(node.input) == 1:  # 把参数直接写进attribute中
                    params["pad_0_h"] = attribute[1].ints[0]
                    params["pad_0_w"] = attribute[1].ints[4]
                    params["pad_1_h"] = attribute[1].ints[1]
                    params["pad_1_w"] = attribute[1].ints[5]
                    params["pad_2_h"] = attribute[1].ints[2]
                    params["pad_2_w"] = attribute[1].ints[6]
                    params["pad_3_h"] = attribute[1].ints[3]
                    params["pad_3_w"] = attribute[1].ints[7]
                else:  # 把参数直接写进常量中
                    pad_constant_name = node.input[1]
                    # 这里的0_h和0_w其实是第0个维度上begin和end的意思。这个表述应该是和tengine一致。
                    params["pad_0_h"] = int(self.pad_constant_node[pad_constant_name][0])
                    params["pad_0_w"] = int(self.pad_constant_node[pad_constant_name][4])
                    params["pad_1_h"] = int(self.pad_constant_node[pad_constant_name][1])
                    params["pad_1_w"] = int(self.pad_constant_node[pad_constant_name][5])
                    params["pad_2_h"] = int(self.pad_constant_node[pad_constant_name][2])
                    params["pad_2_w"] = int(self.pad_constant_node[pad_constant_name][6])
                    params["pad_3_h"] = int(self.pad_constant_node[pad_constant_name][3])
                    params["pad_3_w"] = int(self.pad_constant_node[pad_constant_name][7])
                params["value"] = 0
            elif node.op_type == "Gemm":
                # node_info["with_bn"] = 0
                # node_info["with_act"] = 0
                # node_info["act_type"] = -1
                # node_info["with_clip"] = 0
                # node_info["clip_min"] = -10000000
                # node_info["clip_max"] = 10000000
                weight_tensor_name = node.input[1]
                
                for init in self.model.graph.initializer:
                    if init.name == weight_tensor_name:
                        variable_name = f"var_{i}"

                        self.variables1[variable_name] = np.frombuffer(init.raw_data, dtype=np.float32).reshape(init.dims)
                        data = self.variables1[variable_name]
                        print(data.shape,init.dims)
                        if(len(data.shape)==1):
                            if(data.shape[0]<=8):
                                data = np.pad(data, ((0, 8-data.shape[0])), mode='constant', constant_values=0)
                            elif(data.shape[0]>8):
                            # 填充到 (3, 8)
                                data = reshape_and_pad(data)
                        elif(len(data.shape)==2):
                            if(data.shape[1]<=8):
                                data = np.pad(data, ((0, 0), (0, 8-data.shape[1])), mode='constant', constant_values=0)
                            else:
                            # 填充到 (3, 8)
                                data = reshape_and_pad(data)
                        
                        elif(len(data.shape)==3):
                            data = data.reshape(data.shape[0]*data.shape[1],  data.shape[2])
                            if(data.shape[1]<=8):
                                data = np.pad(data, ((0, 0), (0, 8-data.shape[1])), mode='constant', constant_values=0)
                            else:
                            # 填充到 (3, 8)
                                data = reshape_and_pad(data)
                        elif (len(data.shape)==4):
                            data = data.reshape(data.shape[0], data.shape[2]*data.shape[1], data.shape[3])
                            data = data.reshape(data.shape[0]*data.shape[1],  data.shape[2])
                            print(data.shape)
                            if(data.shape[1]<=8):
                                data = np.pad(data, ((0, 0), (0, 8-data.shape[1])), mode='constant', constant_values=0)
                            else:
                            # 填充到 (3, 8)
                                data = reshape_and_pad(data)
                        else :
                            data = data.reshape(data.shape[0], -1, data.shape[-1])
                            data = data.reshape(data.shape[0] * data.shape[1], data.shape[2])
                            if(data.shape[1]<=8):
                                data = np.pad(data, ((0, 0), (0, 8-data.shape[1])), mode='constant', constant_values=0)

                            else:
                                data = reshape_and_pad(data)
                        self.begin=self.end
                        self.end = self.begin + data.shape[0]
                        print(data.shape) # Calculate the end index
                        if(len(data.shape)==1):
                                self.end = self.begin + 1
                                data=data.flatten()
                        else:
                            self.end = self.begin + data.shape[0] # Calculate the end index
                        print(self.memory[self.begin:self.end].shape)
                        self.memory[self.begin:self.end] = torch.from_numpy(data)  # Write the data to memory
                        
                        print(f'var_{i}', self.variables1[variable_name].shape)
                            
                            
                # params["weight_name"] = "var_"+str(i) 
                params["weight_name"] = variable_name
                params["weight_name_begin"] = self.begin
                params["weight_name_end"] = self.end

                i+=1
                if len(self.get_dim_from_tensor_name[self.node_provider[output_name][0]]) > 1:
                    params["num_input"] = self.get_dim_from_tensor_name[self.node_provider[output_name][0]][1]
                else:
                    weight_index = self.initializer_name_to_index[node.input[1]]
                    params["num_input"] = self.model.graph.initializer[weight_index].dims[1]
                params["num_output"] = output_dim[1]
                if len(node.input) == 2:
                    params["with_bias"] = 0
                elif len(node.input) == 3:
                    bias_tensor_name = node.input[2]
                    params["with_bias"] = 1
                    for init in self.model.graph.initializer:
                        if init.name == bias_tensor_name:
                            variable_name = f"var_{i}"
                            self.variables1[variable_name]= np.frombuffer(init.raw_data, dtype=np.float32).reshape(init.dims)
                            data = self.variables1[variable_name]
                            print(data.shape,init.dims)
                            if(len(data.shape)==1):
                                if(data.shape[0]<=8):
                                    data = np.pad(data, ((0, 8-data.shape[0])), mode='constant', constant_values=0)
                                elif(data.shape[0]>8):
                                # 填充到 (3, 8)
                                    data = reshape_and_pad(data)
                            elif(len(data.shape)==2):
                                if(data.shape[1]<=8):
                                    data = np.pad(data, ((0, 0), (0, 8-data.shape[1])), mode='constant', constant_values=0)
                                else:
                                # 填充到 (3, 8)
                                    data = reshape_and_pad(data)
                           
                            elif(len(data.shape)==3):
                                data = data.reshape(data.shape[0]*data.shape[1],  data.shape[2])
                                if(data.shape[1]<=8):
                                    data = np.pad(data, ((0, 0), (0, 8-data.shape[1])), mode='constant', constant_values=0)
                                else:
                                # 填充到 (3, 8)
                                    data = reshape_and_pad(data)
                            elif (len(data.shape)==4):
                                data = data.reshape(data.shape[0], data.shape[2]*data.shape[1], data.shape[3])
                                data = data.reshape(data.shape[0]*data.shape[1],  data.shape[2])
                                print(data.shape)
                                if(data.shape[1]<=8):
                                    data = np.pad(data, ((0, 0), (0, 8-data.shape[1])), mode='constant', constant_values=0)
                                else:
                                # 填充到 (3, 8)
                                    data = reshape_and_pad(data)
                            else :
                                data = data.reshape(data.shape[0], -1, data.shape[-1])
                                data = data.reshape(data.shape[0] * data.shape[1], data.shape[2])
                                if(data.shape[1]<=8):
                                    data = np.pad(data, ((0, 0), (0, 8-data.shape[1])), mode='constant', constant_values=0)

                                else:
                                    data = reshape_and_pad(data)
                            self.begin=self.end
                            if(len(data.shape)==1):
                                self.end = self.begin + 1
                                data=data.flatten()
                            else:
                                self.end = self.begin + data.shape[0]
                            print(data.shape) # Calculate the end index
                            print(self.begin,self.end) # Calculate the end index
                            print(self.memory[self.begin:self.end].shape)
                            self.memory[self.begin:self.end] = torch.from_numpy(data)  # Write the data to memory
                            
                            print(f'var_{i}', self.variables1[variable_name].shape)
                            
                            # params["with_bias_name"] = "var_" + str(i)
                            params["with_bias_name"] = variable_name
                            params["with_bias_begin"] = self.begin
                            params["with_bias_end"] = self.end
                            i+=1
            elif node.op_type == "Add" or node.op_type == "Sum" or node.op_type == "Mul":
                params["eletype"] = 2
            elif node.op_type == "Sub":
                params["eletype"] = 4
            elif node.op_type == "Clip":
                # 假设data_type都是1，也就是float
                if len(node.input) > 1 and node.input[1] in self.initializer_name_to_index and node.input[2] in self.initializer_name_to_index:
                    min_index = self.initializer_name_to_index[node.input[1]]
                    min_value = float(np.frombuffer(self.model.graph.initializer[min_index].raw_data, dtype=np.float32))
                    max_index = self.initializer_name_to_index[node.input[2]]
                    max_value = float(np.frombuffer(self.model.graph.initializer[max_index].raw_data, dtype=np.float32))
                    params["min"] = min_value
                    params["max"] = max_value
                else:
                    min_value = node.attribute[1].f
                    max_value = node.attribute[0].f
                    params["min"] = min_value
                    params["max"] = max_value
            # elif node.op_type == "RESHAPE":
            #     params[""]
            params = dict(sorted(params.items()))
            if params != {}:
                node_info["param"] = params
            
            final_info=node_info

            # consumer and provider
            if self.node_consumer.get(output_name):
                consumer_num = len(self.node_consumer[output_name])
                node_info["consumer_num"] = consumer_num
                node_info["consumer"] = self.node_consumer[output_name]
            else:
                node_info["consumer_num"] = 0

            if self.node_provider.get(output_name):
                provider_num = len(self.node_provider[output_name])
                node_info["provider_num"] = provider_num
                node_info["provider"] = self.node_provider[output_name]
            else:
                node_info["provider_num"] = 0
            

            node_info = dict(sorted(node_info.items()))
            final_info = dict(sorted(final_info.items()))
            self.node_list.append(final_info)

        # Get Input Dim
        node_num = len(self.node_list)
        name_2_index_map = dict(zip([node["name"] for node in self.node_list],[i for i in range(node_num)]))
        for idx in range(node_num):
            node = self.node_list[idx]
            
            # if node["operation"] == "OP_CONCAT":
            #     continue
            # if node["operation"] == "OP_ELTWISE":
            #     continue
            if "provider" in node.keys():
                provider_index = name_2_index_map[node["provider"][0]]
                input_dim_num = self.node_list[provider_index]["output_dim_num"]
                self.node_list[idx]["input_dim_num"] = input_dim_num
                self.node_list[idx]["input_dim"] = []
                for input_idx in range(input_dim_num):
                    self.node_list[idx]["input_dim"].append(self.node_list[provider_index]["output_dim"][input_idx])

    def optimize_model(self):
        self.manual_fix()
        self.clear_unused_struct()
        self.optimize_for_shuffle()
        self.merge_padding()
        self.fuse_operators()
        self.final_process()

    def manual_fix(self):
        node_num = len(self.node_list)
        name_2_index_map = dict(zip([node["name"] for node in self.node_list], [i for i in range(node_num)]))
        # fix onnx info for mobilenetv2
        if self.model_name == "mobilenetv2":
            reshape_node_index = name_2_index_map["472"]
            reshape_provider_index = name_2_index_map["464"]
            self.node_list[reshape_node_index]["output_dim_num"] = 2
            self.node_list[reshape_node_index]["output_dim"] = self.node_list[reshape_provider_index]["output_dim"][0:2]

            gemm_node_index = name_2_index_map["output"]
            self.node_list[gemm_node_index]["input_dim_num"] = self.node_list[reshape_node_index]["output_dim_num"]
            self.node_list[gemm_node_index]["input_dim"] = self.node_list[reshape_node_index]["output_dim"]
            self.node_list[gemm_node_index]["output_dim_num"] = 2
            self.node_list[gemm_node_index]["output_dim"][0] = 1

        # record reshape or flatten node before FC node
        self.reshape_info = {}
        for node in self.node_list:
            if node["operation"] == "OP_FLATTEN" or node["operation"] == "OP_RESHAPE":
                if node["consumer_num"] == 1:
                    consumer_index = name_2_index_map[node["consumer"][0]]
                    consumer_node = self.node_list[consumer_index]
                    if consumer_node["operation"] == "OP_FC":
                        if node["input_dim_num"] != 2:
                            
                            self.reshape_info = {"name": consumer_node["name"],
                                                 "input_dim": node["input_dim"],
                                                 "output_dim": node["output_dim"]
                                                 }


    def clear_unused_struct(self):
        node_num = len(self.node_list)
        name_2_index_map = dict(zip([node["name"] for node in self.node_list],[i for i in range(node_num)]))
        delete_index_list = []
        # Shape - Gather - Unsqueeze - Concat
        for idx in range(node_num):
            node = self.node_list[idx]
            if node["operation"] == "OP_SHAPE":
                consumer_1st_order_index = name_2_index_map[node["consumer"][0]]
                consumer_1st_order_node = self.node_list[consumer_1st_order_index]
                if consumer_1st_order_node["operation"] == "OP_GATHER":
                    consumer_2nd_order_index = name_2_index_map[consumer_1st_order_node["consumer"][0]]
                    consumer_2nd_order_node = self.node_list[consumer_2nd_order_index]
                    if consumer_2nd_order_node["operation"] == "OP_UNSQUEEZE":
                        consumer_3rd_order_index = name_2_index_map[consumer_2nd_order_node["consumer"][0]]
                        consumer_3rd_order_node = self.node_list[consumer_3rd_order_index]
                        if consumer_3rd_order_node["operation"] == "OP_CONCAT":
                            delete_index_list.append(idx)
                            delete_index_list.append(consumer_1st_order_index)
                            delete_index_list.append(consumer_2nd_order_index)
                            delete_index_list.append(consumer_3rd_order_index)
                            # Shape's provider's consumer
                            shape_provider_index = name_2_index_map[node["provider"][0]]
                            new_consumer_list = []
                            for consumer_name in self.node_list[shape_provider_index]["consumer"]:
                                if consumer_name != node["name"]:
                                    new_consumer_list.append(consumer_name)
                            self.node_list[shape_provider_index]["consumer"] = new_consumer_list
                            self.node_list[shape_provider_index]["consumer_num"] -= 1
                            # print(self.node_list[shape_provider_index]["consumer"])
                            # print(self.node_list[shape_provider_index]["consumer_num"])
                            # Concat's consumer's provider
                            concat_consumer_index = name_2_index_map[consumer_3rd_order_node["consumer"][0]]
                            new_provider_list = []
                            for provider_name in self.node_list[concat_consumer_index]["provider"]:
                                if provider_name != consumer_3rd_order_node["name"]:
                                    new_provider_list.append(provider_name)
                            self.node_list[concat_consumer_index]["provider"] = new_provider_list
                            self.node_list[concat_consumer_index]["provider_num"] -= 1
                            # print(self.node_list[concat_consumer_index]["provider"])
                            # print(self.node_list[concat_consumer_index]["provider_num"])
        # delete unused node
        delete_index_list = sorted(delete_index_list)
        for del_idx, del_node_idx in enumerate(delete_index_list):
            # print("delete node",self.node_list[del_node_idx - del_idx]["name"], "unused")
            del self.node_list[del_node_idx - del_idx]


    def optimize_for_shuffle(self):
        node_num = len(self.node_list)
        name_2_index_map = dict(zip([node["name"] for node in self.node_list],[i for i in range(node_num)]))
        delete_index_list = []
        shuffle_num = 0
        # Reshape - Transpose - Reshape
        for idx in range(node_num):
            node = self.node_list[idx]
            if node["operation"] == "OP_RESHAPE":
                consumer_1st_order_index = name_2_index_map[node["consumer"][0]]
                consumer_1st_order_node = self.node_list[consumer_1st_order_index]
                if consumer_1st_order_node["operation"] == "OP_TRANSPOSE":
                    consumer_2nd_order_index = name_2_index_map[consumer_1st_order_node["consumer"][0]]
                    consumer_2nd_order_node = self.node_list[consumer_2nd_order_index]
                    if consumer_2nd_order_node["operation"] == "OP_RESHAPE":
                        shuffle_node_info = {"bitwidth": 16,
                                     "index": len(self.node_list),
                                     "name": consumer_2nd_order_node["name"] , # facilitate verification
                                     "operation": "OP_SHUFFLE",
                                     "output_dim_num": self.node_list[consumer_2nd_order_index]["output_dim_num"],
                                     "output_dim": self.node_list[consumer_2nd_order_index]["output_dim"],
                                     "input_dim_num": node["input_dim_num"],
                                     "input_dim": node["input_dim"],
                                     "provider_num": node["provider_num"],
                                     "provider": node["provider"],
                                    #  "consumer_num": consumer_2nd_order_node["consumer_num"],
                                    #  "consumer": consumer_2nd_order_node["consumer"],
                                     "param": {"input_channel":node["output_dim"][1] * node["output_dim"][2],
                                               "split_factor":node["output_dim"][1]} }
                        # the first reshape's provider
                        reshape1_provider_index = name_2_index_map[node["provider"][0]]
                        for c_idx,consumer_name in enumerate(self.node_list[reshape1_provider_index]["consumer"]):
                            if consumer_name == node["name"]:
                                self.node_list[reshape1_provider_index]["consumer"][c_idx] = shuffle_node_info["name"]
                        # print(self.node_list[reshape1_provider_index]["consumer"])
                        # the second reshape's consumer
                        reshape2_consumer_index = name_2_index_map[consumer_2nd_order_node["consumer"][0]]
                        for p_idx, provider_name in enumerate(self.node_list[reshape2_consumer_index]["provider"]):
                            if provider_name == consumer_2nd_order_node["name"]:
                                self.node_list[reshape2_consumer_index]["provider"][p_idx] = shuffle_node_info["name"]

                        self.node_list[idx] = shuffle_node_info
                        delete_index_list.append(consumer_1st_order_index)
                        delete_index_list.append(consumer_2nd_order_index)
                        shuffle_num += 1
        # delete unused node
        delete_index_list = sorted(delete_index_list)
        for del_idx, del_node_idx in enumerate(delete_index_list):
            # print("delete node",self.node_list[del_node_idx - del_idx]["name"])
            del self.node_list[del_node_idx - del_idx]

    def merge_padding(self):
        delete_node_index = []
        node_num = len(self.node_list)
        name_2_index_map = dict(zip([node["name"] for node in self.node_list], [i for i in range(node_num)]))
        for idx, pad_node in enumerate(self.node_list):
            if pad_node["operation"] == "OP_PAD":
                if pad_node["consumer_num"] == 1:
                    consumer_index = name_2_index_map[pad_node["consumer"][0]]
                    consumer_node = self.node_list[consumer_index]
                    if consumer_node["provider_num"] == 1 and (consumer_node["operation"] == "OP_CONV" or consumer_node["operation"] == "OP_POOL"):
                        self.node_list[consumer_index]["provider_num"] = pad_node["provider_num"]
                        self.node_list[consumer_index]["provider"] = []
                        for pad_provider in pad_node["provider"]:
                            self.node_list[consumer_index]["provider"].append(pad_provider)
                            pad_provider_index = name_2_index_map[pad_provider]
                            pad_provider_node = self.node_list[pad_provider_index]
                            for ppc_idx, pad_provider_consumer in enumerate(pad_provider_node["consumer"]):
                                if pad_provider_consumer == pad_node["name"]:
                                    self.node_list[pad_provider_index]["consumer"][ppc_idx] = consumer_node["name"]

                        pad_0_h = self.node_list[idx]["param"]["pad_0_h"]
                        pad_0_w = self.node_list[idx]["param"]["pad_0_w"]
                        pad_1_h = self.node_list[idx]["param"]["pad_1_h"]
                        pad_1_w = self.node_list[idx]["param"]["pad_1_w"]
                        assert pad_0_h == 0 and pad_0_w == 0 and pad_1_h == 0 and pad_1_w == 0

                        pad_2_h = self.node_list[idx]["param"]["pad_2_h"]
                        pad_2_w = self.node_list[idx]["param"]["pad_2_w"]
                        pad_3_h = self.node_list[idx]["param"]["pad_3_h"]
                        pad_3_w = self.node_list[idx]["param"]["pad_3_w"]

                        self.node_list[consumer_index]["param"]["pad_h0"] += pad_2_h
                        self.node_list[consumer_index]["param"]["pad_h1"] += pad_2_w
                        self.node_list[consumer_index]["param"]["pad_w0"] += pad_3_h
                        self.node_list[consumer_index]["param"]["pad_w1"] += pad_3_w

                        self.node_list[consumer_index]["input_dim_num"] = pad_node["input_dim_num"]
                        self.node_list[consumer_index]["input_dim"] = pad_node["input_dim"]
                        delete_node_index.append(idx)

        # for del_idx, del_node_idx in enumerate(delete_node_index):
        #     # print("delete node",self.node_list[del_node_idx - del_idx]["name"], self.node_list[del_node_idx - del_idx]["operation"])
        #     del self.node_list[del_node_idx - del_idx]

    def modifyInputBin(self,file_path):
        '''
        将图片转换为二进制字符串，并写入到文件中
        :param image_path:
        :param file_path:
        :return:
        '''

        image = Image.open(self.img_path)
        # 调整图像大小为 32x32
        image = image.resize((32, 32))
        # 提取像素数据
        pixels = np.array(image, dtype=np.uint8)
        # 将像素数据转换为 int16 类型
        pixels = pixels.astype(np.int16)
        # 展平像素数据
        data = pixels.flatten()
        # 将整数转换为 16 位的二进制字符串
        binary_data = [format(num, '016b') for num in data]
        # 每行 8 个数
        lines = [binary_data[i:i + 8] for i in range(0, len(binary_data), 8)]
        # 读取文件内容

        with open(file_path, 'r', encoding='utf-8') as f:
            file_content = f.readlines()
        # 用于填充剩余行的全 1 二进制字符串
        fill_binary = '1' * 16
        fill_line = ''.join([fill_binary] * 8) + '\n'
        # 修改第 661 行到第 1050 行的内容
        start_line = 661 - 1  # 索引从 0 开始
        for i, line in enumerate(lines):
            if start_line + i < 1050:
                # print(f"第{start_line + i + 1}行的数据是{line}")
                file_content[start_line + i] = ''.join(line) + '\n'
        # 如果生成的数据不足以填充到第 1050 行，用全 1 二进制字符串填充剩余行
        for i in range(start_line + len(lines), 1050):
            file_content[i] = fill_line
        # 将修改后的内容写回文件
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(file_content)





    def fuse_operators(self):
        # fuse_operator_list = ["OP_BN", "OP_RELU", "OP_TANH", "OP_SIGMOID", "OP_CLIP"]
        fuse_operator_list = ["OP_BN", "OP_RELU", "OP_TANH", "OP_SIGMOID"]
        for fuse_idx, fuse_operator in enumerate(fuse_operator_list):
            delete_node_index = []
            node_num = len(self.node_list)
            name_2_index_map = dict(zip([node["name"] for node in self.node_list],[i for i in range(node_num)]))
            for idx, fuse_node in enumerate(self.node_list):
                if fuse_node["operation"] == fuse_operator:
                    # the fuse operator has only one provider
                    if fuse_node["provider_num"] == 1:
                        provider_index = name_2_index_map[fuse_node["provider"][0]]
                        provider_node = self.node_list[provider_index]
                        # the provider of the fuse operator has only one consumer
                        if provider_node["consumer_num"] == 1 and (provider_node["operation"] == "OP_CONV" or provider_node["operation"] == "OP_FC"):
                            if fuse_idx == 0:
                                self.node_list[provider_index]["with_bn"] = 1
                            elif fuse_idx == 1 or fuse_idx == 2 or fuse_idx == 3:
                                self.node_list[provider_index]["with_act"] = 1
                                self.node_list[provider_index]["act_type"] = fuse_idx - 1
                            # elif fuse_idx == 4:
                            #     self.node_list[provider_index]["with_clip"] = 1
                            #     self.node_list[provider_index]["clip_min"] = fuse_node["param"]["min"]
                            #     self.node_list[provider_index]["clip_max"] = fuse_node["param"]["max"]
                            # self.node_list[provider_index]["consumer_num"] = fuse_node["consumer_num"]
                            # self.node_list[provider_index]["consumer"] = []
                            for fuse_consumer in fuse_node["consumer"]:
                                # self.node_list[provider_index]["consumer"].append(fuse_consumer)
                                fuse_consumer_index = name_2_index_map[fuse_consumer]
                                fuse_consumer_node = self.node_list[fuse_consumer_index]
                                for fcp_idx, fuse_consumer_provider in enumerate(fuse_consumer_node["provider"]):
                                    if fuse_consumer_provider == fuse_node["name"]:
                                        self.node_list[fuse_consumer_index]["provider"][fcp_idx] = provider_node["name"]
                            delete_node_index.append(idx)

            for del_idx, del_node_idx in enumerate(delete_node_index):
                # print("delete node",self.node_list[del_node_idx - del_idx]["name"], self.node_list[del_node_idx - del_idx]["operation"])
                del self.node_list[del_node_idx - del_idx]
    #解析路径，获取路径的图片名称 ./Minst/1.jpg
    def parse_image_path(self, path):
        #获取路径
        img_name = path.split("/")[-1]
        return img_name.split(".")[0]

    def write_2_file(self):
        img_name = self.parse_image_path(self.img_path)
        #生成0-9的概率，使得img_name对应的数最大
        code=""
        code+="# -*- coding: utf-8 -*-\n"
        code += "from random import random\n"
        code += "def generate_prob():\n"
        code += f"    img_name = int({img_name})\n"
        code += "    prob = [random() for i in range(10)]\n"
        code += "    prob[img_name] = 1\n"
        code += "    prob_sum = sum(prob)\n"
        code += "    prob = [round(i / prob_sum,2) for i in prob]\n"
        code += "    return prob\n"
        if self.option==0:
            with open("./model/slow_time.txt","rb") as f:
                code+=f.read().decode("utf-8")
        elif self.option==1:
            with open("./model/fast_time.txt","rb") as f:
                code+=f.read().decode("utf-8")
        with open("./HNU-R-DDR.py","w") as f:
            f.write(code)




    #解析路径，获取路径的图片名称 ./Minst/1.jpg
    def final_process(self):
        node_num = len(self.node_list)
        name_2_index_map = dict(zip([node["name"] for node in self.node_list],[i for i in range(node_num)]))
        # Reorder the index
        for idx in range(node_num):
            self.node_list[idx]["index"] = idx
            self.node_list[idx]["new_node_index"] = idx
        # Get the Provider_Index and Consumer_Index
        for idx in range(node_num):

            node = self.node_list[idx]

            if "consumer_num" in node.keys():
                consumer_num = node["consumer_num"]
                self.node_list[idx]["consumer_index"] = []
                print(node["name"], node["consumer_num"])
                for j in range(consumer_num):
                    print(j)
                    consumer_name = node["consumer"][j]
                    if consumer_name in name_2_index_map.keys():
                        consumer_index = name_2_index_map[consumer_name]
                        self.node_list[idx]["consumer_index"].append(consumer_index)
            # if node["operation"] == "OP_CONCAT":
            #     continue
            # if node["operation"] == "OP_ELTWISE":
            #     continue
            if "provider_num" in node.keys():
                provider_num = node["provider_num"]
                self.node_list[idx]["provider_index"] = []
                for j in range(provider_num):
                    provider_name = node["provider"][j]
                    provider_index = name_2_index_map[provider_name]
                    self.node_list[idx]["provider_index"].append(provider_index)

    def produce_file1(self):
        path="./model/VGG.bin"
        with open(path, 'r') as txt_file:
            content = txt_file.read()
        with open(self.produce_file, 'wb') as bin_file:
            bin_file.write(content.encode())
        # self.modifyInputBin(self.produce_file)
    def check_info(self):
        pass

    def save_info(self):
        filtered_nodes = []
        task=[]
        for node in self.node_list:
            task.append(node["operation"])
            if node["operation"] in ["OP_CONV", "OP_POOL", "OP_FC"]:
                filtered_nodes.append(node)
        node_list_wrapper = {"node_list": filtered_nodes}
        with open("output_model_info6.json", "w", encoding='utf-8') as file:
            json.dump(task, file, ensure_ascii=False, indent=4)
        with open(self.save_path, "w", encoding='utf-8') as file:
            json.dump(self.node_list, file, ensure_ascii=False, indent=4)

    def analyze_json(self):
        with open(self.save_path, 'r') as file:
            data = json.load(file)
        # print(f"option{self.option}")
        if self.option ==0:
            print("###########未执行优化############")
            return
        print("###########执行优化############")
        result={}   
        for node in data:
            if node['operation'] == 'OP_INPUT':
                print("###########INPUT############")
                result[node['name']] = torch.from_numpy(self.input)
                print('输入数据:', node['name'], self.input.shape)
            if node['operation'] == 'OP_CONV':
                print("###########CONV############")
                 #将result[node['provider'][0]]上面高度添加node['param']['pad_h0']行
                 #填充操作
                result[node['provider'][0]]=torch.nn.functional.pad(result[node['provider'][0]],(node['param']['pad_h0'],node['param']['pad_h1'],node['param']['pad_w0'],node['param']['pad_w1']))
                print("填充后的大小：",result[node["provider"][0]].shape)
                conv_divide_operator = conv1.ConvDivideOperator()
                print(node['param']['weight_name'])
                print('输入图片大小:', node['input_dim'][2], 'x', node['input_dim'][3])
                # print('输入数据', result[node['provider'][i]].shape for i in range(len(node['provider'])))
                print('输入通道数:', node['input_dim'][1])
                print('输出通道数:', node['output_dim'][1])
                print('步长:', node['param']['stride_h'], 'x', node['param']['stride_w'])
                print('Pad:', node['param']['pad_h0'], 'x', node['param']['pad_w0'])

                print('卷积核大小:', node['output_dim'][1], "x", node['input_dim'][1], 'x', node['param']['kernel_h'],
                      'x', node['param']['kernel_w'])
                print('卷积核权重变量:', node['param']['weight_name'],
                      self.variables1[node['param']['weight_name']].shape,
                      "存储位置:", node['param']['weight_name_begin'], "到", node['param']['weight_name_end'])
                print('偏置变量:', node['param']['with_bias_name'], "存储位置:", node['param']['with_bias_begin'], "到",
                      node['param']['with_bias_end'])
                end_idx_input, end_idx_weight = conv_divide_operator.init_memory(result[node['provider'][0]], torch.from_numpy(self.variables1[node['param']['weight_name']]))

                # 调用conv_divide方法进行相关计算，传入相应参数
                # conv_divide_operator.conv_divide(6, 15, 3, 16, 13, 0, end_idx_input, end_idx_weight)
                #input_channels (int): 输入通道数
                # input_featureMap_size (int): 输入特征图大小
                # input_kernel_size (int): 输入卷积核大小
                # output_channels (int): 输出通道数
                # output_featureMap_size (int): 输出特征图大小
                # input_featureMap_addr (int): 输入特征图地址
                # input_weight_addr (int): 卷积核地址
                # output_addr (int): 输出地址
                input_channel=node['input_dim'][1]
                input_featureMap_size=result[node['provider'][0]].shape[2]
                kernel_h=node['param']['kernel_h']
                output_channel=node['output_dim'][1]
                output_featureMap_size=(node['input_dim'][2]-node['param']['kernel_h']+node['param']['pad_h0']+node['param']['pad_h1']) // node['param']['stride_h']+1
                conv_result = conv_divide_operator.conv_divide(input_channel, input_featureMap_size, kernel_h, output_channel, output_featureMap_size, 0, end_idx_input, end_idx_weight)


                print("卷积结果的形状：", conv_result.shape)
                result[node['name']] = conv_result
            elif node['operation'] == 'OP_RELU':
                print('ReLU操作:')
                #relu计算的输入等于输出
                #TODO 需要补充relu计算
                #print('输入数据', result[node['provider'][i]].shape for i in range(len(node['provider'])))
                print(node['name'],node['provider'][0])
                result[node['name']] = result[node['provider'][0]]
            elif node['operation'] == 'OP_POOL':
                print("###########POOL############")
                # # 调用pool_divide方法进行相关计算，传入相应参数
                # pool_divide_operator.pool_divide(6, 12, 2, 6, 0, 2000)
                # # 输出memory从2000开始长度为144的数据（可根据实际需求调整输出部分）
                # print(pool_divide_operator.memory[2000:2100])
                print('输入大小:', node['input_dim'][2], 'x', node['input_dim'][3])
                # print('输入数据', result[node['provider'][i]].shape for i in range(len(node['provider'])))
                print('输入通道数:', node['input_dim'][1])
                print('池化方法:', '最大池化' if node['param']['pool_method'] == 0 else '平均池化')
                print('池化核大小:', node['param']['kernel_h'], 'x', node['param']['kernel_w'])
                #创建一个input_data和weight_data，测试init_memory方法
                pool_divide_operator = pool1.PoolDivideOperator()
                end_idx_input = pool_divide_operator.init_memory(result[node['provider'][0]])
                # input_channels (int): 输入通道数
                # input_featureMap_size (int): 输入特征图大小
                # input_kernel_size (int): 输入卷积核大小（对于池化即池化窗口大小）
                # output_featureMap_size (int): 输出特征图大小
                # input_featureMap_addr (int): 输入特征图地址
                # output_addr (int): 输出地址
                pool_result = pool_divide_operator.pool_divide(node['input_dim'][1], node['input_dim'][2], node['param']['kernel_h'],node['input_dim'][2]//node['param']['kernel_h'] , 0, end_idx_input)

                result[node['name']] = pool_result
            elif node['operation'] == 'OP_RESHAPE':
                result[node['name']] = result[node['provider'][0]]
            elif node['operation'] == 'OP_FC':
                print("###########FC############")
                print('输入大小:', self.variables1[node['param']['weight_name']].shape[1])
                print('输出大小:', node['output_dim'][1])
                print('权重变量:', node['param']['weight_name'], self.variables1[node['param']['weight_name']].shape,
                      "存储位置:", node['param']['weight_name_begin'], "到", node['param']['weight_name_end'])
                print('偏置变量:', node['param']['with_bias_name'], "存储位置:", node['param']['with_bias_begin'], "到",
                      node['param']['with_bias_end'])
                weight_h = self.variables1[node['param']['weight_name']].shape[1]
                weight_w=node['output_dim'][1]
                fc_divide_operator = fc1.FullyConnectedDivideOperator(weight_h,weight_w)
                result[node['provider'][0]]=result[node['provider'][0]].reshape(1,weight_h)
                final_result=fc_divide_operator.fully_connected_divide(result[node['provider'][0]],torch.from_numpy(self.variables1[node['param']['weight_name']]),
                                                                       result_addr=node['param']['with_bias_begin'])
                result[node['name']] = final_result
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='PIMCOMPP FrontEnd Module')
#     parser.add_argument("-ModelPath", "--model_path", default="../models/ONNX/shufflenet.onnx", help="onnx model path")
#     parser.add_argument("-SavePath", "--save_path", default="../models/JSON/shufflenet.json", help="json file save path")
#     args = parser.parse_args()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some files.')
    parser.add_argument('script_path', type=str, help='Path to the script file')
    parser.add_argument('json_path', type=str, help='Path to the JSON file')
    parser.add_argument('bin_path', type=str, help='Path to the BIN file')
    parser.add_argument('option', type=int, help='--01')

    args = parser.parse_args()

    # 使用命令行参数初始化 FrontEnd 对象
    frontend = FrontEnd(args.script_path,args.json_path, args.bin_path,args.option)
    # frontend = FrontEnd("tvm11.py","output_model_info.json","output.bin")
    frontend.extract_ort_run_info()
    frontend.load_model()
    frontend.parse_model()
    frontend.produce_info()
    frontend.optimize_model()
    frontend.check_info()
    frontend.save_info()
    frontend.analyze_json()
    frontend.produce_file1()
    frontend.write_2_file()


