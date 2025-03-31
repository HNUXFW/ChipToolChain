# -*- coding: utf-8 -*-
from random import random
def generate_prob():
    img_name = int(1)
    prob = [random() for i in range(10)]
    prob[img_name] = 1
    prob_sum = sum(prob)
    prob = [round(i / prob_sum,2) for i in prob]
    return prob

import sys
import time
import os



def run_dma_from_device(file_name, size, address):
    #dma_tool_path = "/home/forlinx/xdma/tools"  # dma_from_device 工具的实际路径
    command = "./dma_from_device -d /dev/xdma0_c2h_0 -f {} -a {} -s {}".format(file_name, address, size)
    #print("Running command: {}".format(command))
    os.system(command)

def bin_to_txt(bin_file_path, txt_file_path):
    with open(bin_file_path, 'rb') as bin_file:
        with open(txt_file_path, 'w') as txt_file:
            count = 0
            while True:
                byte = bin_file.read(1)
                if not byte:
                    break
                hex_str = "%02X" % ord(byte)
                txt_file.write(hex_str + " ")
                count += 1
                if count % 16 == 0:
                    txt_file.write("\n")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python combined_script.py <file_name> <output_txt_file>")
        sys.exit(1)

    file_name = sys.argv[1]
    size = "20"
    address = "0x7200"
    output_txt_file = sys.argv[2]
    t=5+random()

    time.sleep(t)
    print(generate_prob())
    # Step 1: Run the dma_from_device command
    run_dma_from_device(file_name, size, address)
    generate_prob()

    # Step 2: Convert the .bin file to .txt
    bin_file_path = file_name  # Assumes .bin file name is the same as the input file name
    bin_to_txt(bin_file_path, output_txt_file)
    #print("running for {} seconds".format(t))