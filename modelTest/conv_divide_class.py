import torch
import torch.nn as nn
import numpy


class ConvDivideOperator:
    def __init__(self, memory_size=1000000):
        """
        初始化类，创建指定大小的内存空间，并初始化算子库。

        参数:
        memory_size (int): 内存大小，默认为1000000
        """
        self.memory_size = memory_size
        self.memory = torch.zeros(self.memory_size)
        self.operator_library = {}
        self._init_operator_library()

    def _init_operator_library(self):
        """
        初始化算子库，向其中添加模拟的算子信息。
        """
        # 算子1
        operator1_name = "conv1"
        operator1_info = {
            "input_channels": 6,
            "input_feature_map_size": (6, 6),
            "convolution_kernel_size": (3, 3),
            "operator_type": "conv",
            "stride": 1
        }
        self.operator_library[operator1_name] = operator1_info


        # 算子2
        operator2_name = "conv2"
        operator2_info = {
            "input_channels": 3,
            "input_feature_map_size": (8, 8),
            "convolution_kernel_size": (3, 3),
            "operator_type": "conv",
            "stride": 1
        }
        self.operator_library[operator2_name] = operator2_info

        # 算子1
        operator3_name = "conv3"
        operator3_info = {
            "input_channels": 1,
            "input_feature_map_size": (8, 8),
            "convolution_kernel_size": (3, 3),
            "operator_type": "conv",
            "stride": 1
        }
        self.operator_library[operator3_name] = operator3_info

        operator4_name = "conv4"
        operator4_info = {
            "input_channels": 6,
            "input_feature_map_size": (15, 15),
            "convolution_kernel_size": (3, 3),
            "operator_type": "conv",
            "stride": 1
        }
        self.operator_library[operator4_name] = operator4_info

        # 算子3
        operator5_name = "maxpool"
        operator5_info = {
            "input_channels": 6,
            "input_feature_map_size": (6, 6),
            "convolution_kernel_size": (3, 3),  # 对于池化操作这里可理解为池化窗口大小
            "operator_type": "pool",
            "stride": 2
        }
        self.operator_library[operator5_name] = operator5_info

        # 算子1
        operator6_name = "conv6"
        operator6_info = {
            "input_channels": 3,
            "input_feature_map_size": (33,33),
            "convolution_kernel_size": (3, 3),
            "operator_type": "conv",
            "stride" : 2
        }
        self.operator_library[operator6_name] = operator6_info

        operator7_name = "conv7"
        operator7_info = {
            "input_channels": 4,
            "input_feature_map_size": (17,17),
            "convolution_kernel_size": (3, 3),
            "operator_type": "conv",
            "stride" : 1
        }
        self.operator_library[operator7_name] = operator7_info

        operator8_name = "conv8"
        operator8_info = {
            "input_channels": 4,
            "input_feature_map_size": (18,18),
            "convolution_kernel_size": (3, 3),
            "operator_type": "conv",
            "stride" : 1
        }
        self.operator_library[operator8_name] = operator8_info

        operator9_name = "conv9"
        operator9_info = {
            "input_channels": 4,
            "input_feature_map_size": (10,10),
            "convolution_kernel_size": (3, 3),
            "operator_type": "conv",
            "stride" : 1
        }
        self.operator_library[operator9_name] = operator9_info

        operator10_name = "conv10"
        operator10_info = {
            "input_channels": 8,
            "input_feature_map_size": (10,10),
            "convolution_kernel_size": (3, 3),
            "operator_type": "conv",
            "stride" : 1
        }
        self.operator_library[operator10_name] = operator10_info

        operator11_name = "conv11"
        operator11_info = {
            "input_channels": 8,
            "input_feature_map_size": (6,6),
            "convolution_kernel_size": (3, 3),
            "operator_type": "conv",
            "stride" : 1
        }
        self.operator_library[operator11_name] = operator11_info

        operator12_name = "conv12"
        operator12_info = {
            "input_channels": 16,
            "input_feature_map_size": (6,6),
            "convolution_kernel_size": (3, 3),
            "operator_type": "conv",
            "stride" : 1
        }
        self.operator_library[operator12_name] = operator12_info

        operator13_name = "conv13"
        operator13_info = {
            "input_channels": 16,
            "input_feature_map_size": (4,4),
            "convolution_kernel_size": (3, 3),
            "operator_type": "conv",
            "stride" : 1
        }
        self.operator_library[operator13_name] = operator13_info

        operator14_name = "conv14"
        operator14_info = {
            "input_channels": 32,
            "input_feature_map_size": (4,4),
            "convolution_kernel_size": (3, 3),
            "operator_type": "conv",
            "stride" : 1
        }
        self.operator_library[operator14_name] = operator14_info

    def init_memory(self,input_data, weight_data):
        """
        初始化memory里的数据

        参数：
        input_data (张量类型) : 输入数据
        weight_data (张量类型): 权重数据

        返回：
        输入数据的结束地址和权重数据的结束地址
        """
        self.input_data = input_data
        self.weight_data = weight_data

        input_size = input_data.numel()
        weight_size = weight_data.numel()

        if input_size + weight_size > self.memory_size:
            raise ValueError("输入数据和权重数据的总元素个数超过了memory的大小")

        # 将input_data平铺后放入memory
        start_idx = 0
        end_idx_input = start_idx + input_size
        self.memory[start_idx:end_idx_input] = input_data.view(-1)

        # 将weight_data平铺后接着放入memory
        start_idx_weight = end_idx_input
        end_idx_weight = start_idx_weight + weight_size
        self.memory[start_idx_weight:end_idx_weight] = weight_data.view(-1)

        return end_idx_input, end_idx_weight
    
    def find_operator(self, operator_type,input_channels,feature_map_size, kernel_size):
        """
        在算子库中查找符合条件的算子信息。

        参数:
        operator_type (str): 算子类型，取值为 'conv'（卷积）或 'pool'（池化）
        feature_map_size (tuple): 输入特征图大小，格式为 (height, width)
        kernel_size (tuple): 卷积核大小或者池化窗口大小，格式为 (kernel_height, kernel_width)

        返回:
        dict or None: 符合条件的算子信息字典，若没找到则返回None。
        """
        print("-----寻找匹配算子-----")
        for operator_name, operator_info in self.operator_library.items():
            stored_input_channels = operator_info["input_channels"]
            stored_kernel_size = operator_info["convolution_kernel_size"]
            stored_feature_map_size = operator_info["input_feature_map_size"]
            stored_operator_type = operator_info["operator_type"]
            stored_stride = operator_info["stride"]

            # 判断算子类型是否匹配
            if stored_operator_type == operator_type and stored_input_channels == input_channels:
                # 判断卷积核大小或者池化窗口大小是否匹配
                if stored_kernel_size == kernel_size:
                    height_diff = feature_map_size[0] - stored_feature_map_size[0]
                    divisor_height = stored_feature_map_size[0] - (stored_kernel_size[0] - 1)
                    if divisor_height!= 0 and height_diff % divisor_height == 0:
                        block_nums = (height_diff // divisor_height + 1)
                        return operator_info, block_nums
        return None

    def get_addresses(self, addr):
        """
        根据给定地址获取对应通道数的子图首地址。

        参数:
        addr (int): 起始地址

        返回:
        list: 地址列表
        """
        addresses = []
        j = 0
        for i in range(addr, len(self.memory), self.input_featureMap_size * self.input_featureMap_size):
            if j == self.input_channels:
                break
            addresses.append(i)
            j += 1
        return addresses

    def get_outputAddresses(self, start_output_address, block_num, output_featureMap_size, operator_output_size):
        """
        计算并获取输出的地址信息（二维数组形式）。

        参数:
        start_output_address (int): 输出起始地址
        block_num (int): 分块编号
        output_featureMap_size (int): 输出特征图大小
        operator_output_size (int): 算子输出大小

        返回:
        list: 二维地址列表
        """
        output_size = output_featureMap_size
        sub_output_size = operator_output_size

        # 计算输出的起始地址
        addr = start_output_address + (block_num // (output_size // sub_output_size)) * (
                output_size * sub_output_size) + (block_num % (output_size // sub_output_size)) * sub_output_size

        outputAddresses = []
        for i in range(sub_output_size):
            outputAddresses.append(addr + i * output_size)

        return outputAddresses

    def create_and_convolve(self, addresses, outputAddresses,operator_input_channels,
                                     operator_input_feature_map_size,operator_convolution_kernel_size,operator_stride,weight_data):
        """          
        模拟执行卷积操作，包括读取数据、创建卷积层、进行卷积运算以及将结果写回内存。

        参数:     
        addresses (list): 输入地址列表
        outputAddresses (list): 输出地址列表
        operator_input_channels: 输入通道数
        operator_input_feature_map_size: 输入特征图大小
        operator_convolution_kernel_size: 卷积核大小
        weight_data: 权重数据


        返回:
        int: 操作执行状态码（当前固定返回1表示成功，可按需修改）
        """
        # 创建一个空的输入大小张量
        input_tensor = torch.zeros(operator_input_channels, operator_input_feature_map_size[0], operator_input_feature_map_size[1])
        print("input_tensor的形状:", input_tensor.shape)

        # 从每个地址读取6x6的数据，并填充到input_tensor中
        for i, addr in enumerate(addresses):
            for row in range(operator_input_feature_map_size[0]):
                for col in range(operator_input_feature_map_size[0]):
                    # 计算在14x14矩阵中的偏移量
                    offset = row * self.input_featureMap_size + col
                    input_tensor[i, row, col] = self.memory[addr + offset]

        # # 将input_tensor写入result.txt文件中
        # with open('/mnt/sdb/yangfan/python/result_new.txt', 'w') as file:
        #     file.write(str(input_tensor))

        # 创建一个卷积层，卷积核大小为3x3，输入通道数为6，输出通道数为1
        conv_layer = nn.Conv2d(in_channels=operator_input_channels, out_channels=1, kernel_size=operator_convolution_kernel_size, stride=operator_stride, padding=0)
        conv_layer.weight.data = weight_data        

        # 对input_tensor进行卷积
        input_tensor = input_tensor.unsqueeze(0)  # 添加批次维度
        output_tensor = conv_layer(input_tensor)  # 进行算子卷积

        # 将output_tensor写入result.txt文件中
        # with open('/mnt/sdb/yangfan/python/result_conv.txt', 'w') as file:
        #     file.write(str(output_tensor))

        # 将output_tensor铺平写入memory中
        flattened_output = output_tensor.view(-1)

        z = 0
        # 按照sub_output_size，逐行输出结果到memory中
        for i in outputAddresses:
            for j in range(len(outputAddresses)):
                self.memory[i + j] = flattened_output[z]
                z += 1
        return 1

    def conv_divide(self, input_channels, input_featureMap_size, input_kernel_size,
                    output_channels, output_featureMap_size,
                    input_featureMap_addr, input_weight_addr, output_addr):
        """
        执行主要的卷积划分操作流程，包括选择算子、分块处理、调用相关函数进行计算等。

        参数:
        input_channels (int): 输入通道数
        input_featureMap_size (int): 输入特征图大小
        input_kernel_size (int): 输入卷积核大小
        output_channels (int): 输出通道数
        output_featureMap_size (int): 输出特征图大小
        input_featureMap_addr (int): 输入特征图地址
        input_weight_addr (int): 卷积核地址
        output_addr (int): 输出地址
        """
        self.input_channels = input_channels
        self.input_featureMap_size = input_featureMap_size
        self.input_kernel_size = input_kernel_size
        self.output_channels = output_channels
        self.output_featureMap_size = output_featureMap_size

        # # 根据输入地址，生成随机的输入数据，将其展平后存入memory的前面部分
        # input_tensor = torch.randint(0, 256, (self.input_channels, self.input_featureMap_size, self.input_featureMap_size),
        #                              dtype=torch.float32)
        # with open('/mnt/sdb/yangfan/python/result.txt', 'w') as file:
        #     file.write(str(input_tensor))
        # flattened_input = input_tensor.view(-1)
        # self.memory[:flattened_input.size(0)] = flattened_input

        # 1、选择算子：补充没找到相应算子的处理
        operator_info, block_nums = self.find_operator("conv", self.input_channels,(self.input_featureMap_size, self.input_featureMap_size),
                                                       (self.input_kernel_size, self.input_kernel_size))
        print("算子信息：", operator_info)
        print("总的分块数:block_nums:", block_nums * block_nums)

        # 根据选用的算子，计算需要的分块数，循环处理每个分块
        operator_input_channels = operator_info["input_channels"]
        operator_input_feature_map_size = operator_info["input_feature_map_size"]
        operator_convolution_kernel_size = operator_info["convolution_kernel_size"]
        operator_operator_type = operator_info["operator_type"]
        operator_stride = operator_info["stride"]
        operator_output_size = (operator_input_feature_map_size[0] - operator_convolution_kernel_size[0] + 1) // operator_stride

        # TODO：加一个输出通道的循环，每个输出通道去拿一个对应的权重传进去
        for i in range(self.output_channels):

            #TODO: 获得此通道对应的权重数据
            channel_weight_data = self.weight_data[i].unsqueeze(0)
            print("channel_weight_data的形状:", channel_weight_data.shape)

            for block_num in range(block_nums ** 2):
                print("---第{}个分块卷积---".format(block_num))
                # 计算输入的地址
                addr = input_featureMap_addr + (block_num // block_nums) * (
                        self.input_featureMap_size * (
                                operator_input_feature_map_size[0] - operator_convolution_kernel_size[0] + 1)) + (
                                block_num % block_nums) * (
                                operator_input_feature_map_size[0] - operator_convolution_kernel_size[0] + 1)

                # 获取输入地址
                inputAddresses = self.get_addresses(addr)


                index = self.output_featureMap_size * self.output_featureMap_size * i
                print("index:", index)
                # 获取输出地址,加上通道数对应的长度
                outputAddresses = self.get_outputAddresses(output_addr + index, block_num, self.output_featureMap_size,
                                                            operator_output_size)

                # 执行卷积及相关操作
                self.create_and_convolve(inputAddresses, outputAddresses,operator_input_channels,
                                        operator_input_feature_map_size,operator_convolution_kernel_size,operator_stride,channel_weight_data)
        
        # 返回卷积的结果
        # memory数组一部分元素reshape成张量
        result = self.memory[output_addr:output_addr + self.output_channels * self.output_featureMap_size * self.output_featureMap_size]
        result = result.reshape(self.output_channels, self.output_featureMap_size, self.output_featureMap_size)
        return result
        
        
        
# # 创建类的实例
# conv_divide_operator = ConvDivideOperator()

# #创建一个input_data和weight_data，测试init_memory方法
# input_data = torch.randint(0, 256, (1, 32, 32), dtype=torch.float32)  
# weight_data = torch.randint(0, 256, (6, 1,3, 3), dtype=torch.float32)
# end_idx_input, end_idx_weight = conv_divide_operator.init_memory(input_data, weight_data)
# print("输入数据的结束地址：", end_idx_input)
# print("权重数据的结束地址：", end_idx_weight)

# # 调用conv_divide方法进行相关计算，传入相应参数
# # conv_divide_operator.conv_divide(6, 15, 3, 16, 13, 0, end_idx_input, end_idx_weight)
# conv_result = conv_divide_operator.conv_divide(1, 32, 3, 6, 30, 0, end_idx_input, end_idx_weight)
# print("卷积结果的形状：", conv_result.shape)
# # 输出memory从2000开始长度为144的数据（可根据实际需求调整输出部分）
# print(conv_divide_operator.memory[2000:2100])