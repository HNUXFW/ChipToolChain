import torch
import torch.nn as nn


class PoolDivideOperator:
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

    def init_memory(self,input_data):
        """
        初始化memory里的数据

        参数：
        input_data (张量类型) : 输入数据

        返回：
        输入数据的结束地址
        """
        input_size = input_data.numel()

        # 将input_data平铺后放入memory
        start_idx = 0
        end_idx_input = start_idx + input_size
        self.memory[start_idx:end_idx_input] = input_data.view(-1)

        return end_idx_input
    
    def _init_operator_library(self):
        """
        初始化算子库，向算子库中添加一些模拟的算子信息。
        """

        # 算子3
        operator3_name = "maxpool1"
        operator3_info = {
            "input_channels": 6,
            "input_feature_map_size": (6, 6),
            "convolution_kernel_size": (2, 2),  # 对于池化操作这里可以理解为池化窗口大小
            "operator_type": "pool"
        }
        self.operator_library[operator3_name] = operator3_info

        # 算子4
        operator4_name = "maxpool2"
        operator4_info = {
            "input_channels": 16,
            "input_feature_map_size": (13, 13),
            "convolution_kernel_size": (2, 2),  # 对于池化操作这里可以理解为池化窗口大小
            "operator_type": "pool"
        }
        self.operator_library[operator4_name] = operator4_info

        # 算子5
        operator5_name = "maxpool3"
        operator5_info = {
            "input_channels": 4,
            "input_feature_map_size": (16, 16),
            "convolution_kernel_size": (2, 2),
            "operator_type": "pool"
        }
        self.operator_library[operator5_name] = operator5_info

        # 算子6
        operator6_name = "maxpool4"
        operator6_info = {
            "input_channels": 8,
            "input_feature_map_size": (8, 8),
            "convolution_kernel_size": (2, 2),
            "operator_type": "pool"
        }
        self.operator_library[operator6_name] = operator6_info

        # 算子7
        operator7_name = "maxpool5"
        operator7_info = {
            "input_channels": 16,
            "input_feature_map_size": (4, 4),
            "convolution_kernel_size": (2, 2),
            "operator_type": "pool"
        }
        self.operator_library[operator7_name] = operator7_info

        # 算子8
        operator8_name = "maxpool6"
        operator8_info = {
            "input_channels": 32,
            "input_feature_map_size": (2, 2),
            "convolution_kernel_size": (2, 2),
            "operator_type": "pool"
        }
        self.operator_library[operator8_name] = operator8_info

    def find_operator(self,  operator_type, input_channels,feature_map_size, kernel_size):
        """
        在算子库中查找符合条件的算子信息。

        参数:
        operator_type (str): 算子类型，取值为 'conv'（卷积）或者 'pool'（池化）
        feature_map_size (tuple): 输入特征图大小，格式为 (height, width)
        kernel_size (tuple): 卷积核大小或者池化窗口大小，格式为 (kernel_height, kernel_width)

        返回:
        dict or None: 符合条件的算子信息字典，如果没有找到符合条件的算子则返回 None。
        """
        print("-----寻找匹配算子-----")
        for operator_name, operator_info in self.operator_library.items():
            stored_input_channels = operator_info["input_channels"]
            stored_kernel_size = operator_info["convolution_kernel_size"]
            stored_feature_map_size = operator_info["input_feature_map_size"]
            stored_operator_type = operator_info["operator_type"]

            # 先判断算子类型是否匹配
            if stored_operator_type == operator_type and stored_input_channels == input_channels:
                # 再判断卷积核大小或者池化窗口大小是否匹配
                if stored_kernel_size == kernel_size:
                    # 输入特征图大小根据公式判断选择算子
                    if feature_map_size[0] % stored_feature_map_size[0] == 0:
                        block_nums = feature_map_size[0] // stored_feature_map_size[0]
                        return operator_info, block_nums
        return None

    def get_addresses(self, addr):
        """
        获得对应通道数的子图首地址。

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
        输出对应通道数的地址（二维数组）。

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

        outputAddresses = []

        for i in range(self.input_channels):
            start_output_address_channel = start_output_address + i * output_size * output_size
            # 计算输出的起始地址
            addr = start_output_address_channel + (block_num // (output_size // sub_output_size)) * (
                    output_size * sub_output_size) + (block_num % (output_size // sub_output_size)) * sub_output_size

            outputAddresses_oneChannel = []
            for i in range(sub_output_size):
                outputAddresses_oneChannel.append(addr + i * output_size)
            outputAddresses.append(outputAddresses_oneChannel)
        return outputAddresses

    def create_and_convolve(self, addresses, outputAddresses,operator_input_channels,
                            operator_input_feature_map_size,operator_convolution_kernel_size):
        """
        模拟的算子操作，包括读取数据、创建张量、进行池化并将结果写回内存。

        参数:
        addresses (list): 输入地址列表
        outputAddresses (list): 输出地址列表

        返回:
        int: 返回1表示操作完成（可根据实际情况调整返回值含义）
        """
        # 创建一个空的6x6x6张量
        input_tensor = torch.zeros(operator_input_channels, operator_input_feature_map_size[0], operator_input_feature_map_size[1])
        print("input_tensor的形状:", input_tensor.shape)

        # 从每个地址读取6x6的数据，并填充到input_tensor中
        for i, addr in enumerate(addresses):
            for row in range(operator_input_feature_map_size[0]):
                for col in range(operator_input_feature_map_size[0]):
                    # 计算在12X12矩阵中的偏移量
                    offset = row * self.input_featureMap_size + col
                    input_tensor[i, row, col] = self.memory[addr + offset]

        # 将input_tensor写入result.txt文件中
        # with open('/mnt/sdb/yangfan/python/pool/result_new.txt', 'w') as file:
        #     file.write(str(input_tensor))

        max_pool = nn.MaxPool2d(kernel_size=operator_convolution_kernel_size)

        # 对input_tensor进行池化
        input_tensor = input_tensor.unsqueeze(0)  # 添加批次维度
        output_tensor = max_pool(input_tensor)
        # 将output_tensor写入result.txt文件中
        # with open('/mnt/sdb/yangfan/python/pool/result_pool.txt', 'w') as file:
        #     file.write(str(output_tensor))

        print("输出的形状：", output_tensor.shape)

        # 遍历通道维度
        for channel_idx in range(output_tensor.shape[1]):
            channel_data = output_tensor[0, channel_idx, :, :]

            flattened_output = channel_data.view(-1)
            z = 0
            # 然后按照sub_output_size，每一行输出结果，再加上output_size输出下一行的数据
            for i in outputAddresses[channel_idx]:
                for j in range(output_tensor.shape[2]):
                    self.memory[i + j] = flattened_output[z]
                    z += 1
        return 1

    def pool_divide(self, input_channels, input_featureMap_size, input_kernel_size,
                    output_featureMap_size,
                    input_featureMap_addr, output_addr):
        """
        执行主要的池化划分操作流程，包括选择算子、分块处理、调用相关函数进行计算等。

        参数:
        input_channels (int): 输入通道数
        input_featureMap_size (int): 输入特征图大小
        input_kernel_size (int): 输入卷积核大小（对于池化即池化窗口大小）
        output_featureMap_size (int): 输出特征图大小
        input_featureMap_addr (int): 输入特征图地址
        output_addr (int): 输出地址

        """
        self.input_channels = input_channels
        self.input_featureMap_size = input_featureMap_size
        self.input_kernel_size = input_kernel_size
        self.output_featureMap_size = output_featureMap_size

        # # 根据输入地址，去访问得到对应的数据，目前只是直接读数据
        # input_tensor = torch.randint(0, 256, (self.input_channels, self.input_featureMap_size, self.input_featureMap_size),
        #                              dtype=torch.float32)
        # # 将input_tensor写入result.txt文件中
        # with open('/mnt/sdb/yangfan/python/pool/result.txt', 'w') as file:
        #     file.write(str(input_tensor))
        # # 将输入数据展平为一维数组
        # flattened_input = input_tensor.view(-1)
        # # 将展平后的输入数据放到memory的前面部分
        # self.memory[:flattened_input.size(0)] = flattened_input

        # 1、选择算子：补充没找到相应算子的处理
        operator_info, block_nums = self.find_operator("pool", self.input_channels,(self.input_featureMap_size, self.input_featureMap_size),
                                                       (self.input_kernel_size, self.input_kernel_size))
        print("算子信息：", operator_info)
        print("总的分块数:block_nums:", block_nums * block_nums)

        # 根据选用的算子，计算需要的分块数
        # 循环去计算完每个分块结果
        # 从operator_info中获取算子的信息，包括计算输出的大小
        operator_input_channels = operator_info["input_channels"]
        operator_input_feature_map_size = operator_info["input_feature_map_size"]
        operator_convolution_kernel_size = operator_info["convolution_kernel_size"]
        operator_operator_type = operator_info["operator_type"]
        operator_output_size = operator_input_feature_map_size[0] // operator_convolution_kernel_size[0]

        # 根据block_nums来进行循环，调用下面的计算
        for block_num in range(block_nums ** 2):
            print("---第{}个分块池化---".format(block_num))

            # 计算输入的地址
            addr = input_featureMap_addr + (block_num // block_nums) * (
                    self.input_featureMap_size * operator_input_feature_map_size[0]) + (
                           block_num % block_nums) * operator_input_feature_map_size[0]
            print("addr:" + str(block_num), addr)

            # Call the function to get the addresses
            inputAddresses = self.get_addresses(addr)
            print(inputAddresses)

            outputAddresses = self.get_outputAddresses(output_addr, block_num, self.output_featureMap_size,
                                                       operator_output_size)
            print(outputAddresses)

            self.create_and_convolve(inputAddresses, outputAddresses, operator_input_channels,
                                     operator_input_feature_map_size, operator_convolution_kernel_size)

        # 返回池化的结果
        # 并reshape成张量
        result= self.memory[output_addr:output_addr + self.input_channels * self.output_featureMap_size * self.output_featureMap_size]
        result = result.reshape(self.input_channels, self.output_featureMap_size, self.output_featureMap_size)
        return result

# 创建类的实例
# pool_divide_operator = PoolDivideOperator()
# # # 调用pool_divide方法进行相关计算，传入相应参数
# # pool_divide_operator.pool_divide(6, 12, 2, 6, 0, 2000)
# # # 输出memory从2000开始长度为144的数据（可根据实际需求调整输出部分）
# # print(pool_divide_operator.memory[2000:2100])

# #创建一个input_data和weight_data，测试init_memory方法
# input_data = torch.randint(0, 256, (6, 30, 30), dtype=torch.float32)  
# end_idx_input = pool_divide_operator.init_memory(input_data)
# print("输入数据的结束地址：", end_idx_input)
# #输出memory从0开始到end_idx_weight的数据
# # print(pool_divide_operator.memory[:end_idx_weight])
# pool_result = pool_divide_operator.pool_divide(6, 30, 2, 15, 0, end_idx_input)
# print("池化结果的形状：", pool_result.shape)
# print(pool_result)  
# print(pool_divide_operator.memory[end_idx_input:end_idx_input+6*15*15])
