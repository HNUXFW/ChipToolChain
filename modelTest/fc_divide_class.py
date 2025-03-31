import torch
import torch.nn as nn

class FullyConnectedDivideOperator:
    def __init__(self,weight_h,weight_w,memory_size=1000000):
        """
        初始化类，创建指定大小的内存空间。

        参数:
        memory_size (int): 内存大小，默认为1000000
        """
        self.memory_size = memory_size
        self.memory = torch.zeros(self.memory_size)
        self.input_size=weight_h
        self.weight_w=weight_w
        self.weight_h=weight_h
        self.operator_library = {}
        self._init_operator_library()


    def _init_operator_library(self):
        """
        初始化算子库，向其中添加模拟的算子信息。
        :return:
        """
        operator1_name = "fc1"
        operator1_info = {
            "block_size": 8,
            "operator_type": "fc"
        }
        self.operator_library[operator1_name] = operator1_info

        operator2_name = "fc2"
        operator2_info = {
            "block_size": 4,
            "operator_type": "fc"
        }
        self.operator_library[operator2_name] = operator2_info

        operator3_name = "fc3"
        operator3_info = {
            "block_size": 2,
            "operator_type": "fc"
        }
        self.operator_library[operator3_name] = operator3_info


    def find_operator(self):
        """
        查找算子，根据输入张量大小和权重张量大小查找对应的算子。
        :return:
        """
        #查看是否输入大小能被输入的窗口大小整除，权重大小能被权重的窗口大小整除。
        print("-----寻找匹配算子-----")
        for operator_name, operator_info in self.operator_library.items():
            if self.input_size % operator_info["block_size"] == 0 and self.weight_w % operator_info["block_size"] == 0:
                return operator_name, operator_info
        return None, None

    def transfer_to_memory(self, input_tensor, address):
        """
        将张量放置在模拟内存中，首先判断维度然后展平。
        参数:
        input_tensor (torch.Tensor): 输入张量
        address (int): 内存起始地址
        """
        if len(input_tensor.size()) == 1:
            self.memory[address:address + input_tensor.size(0)] = input_tensor
        else:
            h, w = input_tensor.size()
            self.memory[address:address +h*w] = input_tensor.flatten()

    def fc_divide(self, input_tensor_addr, weight_tensor_addr, result_tensor_addr, block_size, weight_w):
        """
        分块矩阵乘法，将结果存储到指定内存地址中。

        参数:
        input_tensor_addr (int): 输入张量起始地址
        weight_tensor_addr (int): 权重张量起始地址
        result_tensor_addr (int): 结果张量起始地址
        block_size (int): 分块大小
        weight_w (int): 权重矩阵的宽度
        """
        #获取输入数据
        block_input_tensor = self.memory[input_tensor_addr:input_tensor_addr + block_size]
        block_weight_tensor = torch.zeros(block_size, block_size)
        #获取权重数据
        for idx in range(block_size):
            block_weight_tensor[idx:idx + 1, 0:block_size] = self.memory[weight_tensor_addr + (idx * weight_w):
                                                                        weight_tensor_addr + (idx * weight_w) + block_size].unsqueeze(0)
        block_result_tensor = torch.matmul(block_input_tensor, block_weight_tensor).flatten()
        self.memory[result_tensor_addr:result_tensor_addr + block_size] = block_result_tensor

    def sum_memory(self, middle_result_addr, block_size, block_H_num, block_W_num):
        """
        将归属于同一列的结果进行加法。

        参数:
        middle_result_addr (int): 中间结果起始地址
        block_size (int): 分块大小
        block_H_num (int): 行方向上的块数
        block_W_num (int): 列方向上的块数
        weight_w (int): 权重矩阵的宽度

        返回:
        torch.Tensor: 最终结果张量
        """
        sum_tensor = torch.zeros(self.weight_w)
        for i in range(block_W_num):  # 行方向的块数
            for j in range(block_H_num):  # 列方向上的块数
                block_result_addr = middle_result_addr + block_H_num * i * block_size + j * block_size
                # print("打印此时的sum_tensor:", sum_tensor[i * block_size:i * block_size + block_size])
                # print("打印此时的memory:", self.memory[block_result_addr: block_result_addr + block_size])
                sum_tensor[i * block_size:i * block_size + block_size] += self.memory[block_result_addr: block_result_addr + block_size]
        return sum_tensor

    def fully_connected_divide(self,input_tensor = None,weight_tensor=None,middle_result_addr = 70000,result_addr=80000):
        """
        执行全连接划分操作流程，包括分块处理、调用相关函数进行计算等。
        首先是获取输入的地址和权重的地址，之后将按照分块的方式从中读取，
        获取的各个部分再进行乘法。将中间结果再存放在一个中间地址。
        最后将中间结果进行加法，得到最终的结果。

        参数:
        input_tensor_addr (int): 输入张量起始地址
        weight_tensor_addr (int): 权重张量起始地址
        middle_result_addr (int): 中间结果起始地址
        result_addr (int): 最终结果起始地址
        """
        # 初始化输入张量和权重张量
        input_tensor_addr = 0
        weight_tensor_addr = 5000
        # 将输入张量和权重张量转移到内存中
        self.transfer_to_memory(input_tensor, input_tensor_addr)
        self.transfer_to_memory(weight_tensor, weight_tensor_addr)
        operator_name,operation_info = self.find_operator()
        block_size=operation_info["block_size"]

        print("算子信息:",operation_info)
        # 计算分块数量
        block_H_num = self.weight_h // block_size
        block_W_num = self.weight_w // block_size
        print("输入的分块数量:",block_H_num)
        print(f"权重的分块数量:{block_H_num}*{block_W_num}")
        block_input_num = block_H_num
        # 默认块乘法的存储地址
        # 将输入张量的地址存进列表中
        input_address_list = [input_tensor_addr + i * block_size for i in range(block_input_num)]

        # 将权重张量的地址存进列表中，权重张量的地址是按照从左到右，从上到下的顺序进行
        weight_address_list = []
        for i in range(0, self.weight_h, block_size):
            for j in range(0, self.weight_w, block_size):
                weight_address_list.append(weight_tensor_addr + i * self.weight_w + j)

        # 调用分块矩阵乘法函数，得到的一列矩阵的结果存储到内存。
        for i in range(block_W_num):
            for j in range(block_input_num):
                self.fc_divide(input_address_list[j], weight_address_list[j * block_W_num + i],
                                                         middle_result_addr + (i * block_input_num + j) * block_size, block_size, self.weight_w)

        # 计算最终结果
        final_result = self.sum_memory(middle_result_addr, block_size, block_H_num, block_W_num)
        # 将最终结果存入内存
        self.memory[result_addr:result_addr + self.weight_w] = final_result
        return final_result
