# -*- coding: utf-8 -*-
import sys
import os

def run_dma_to_device(file_name, size):
    command = "./dma_to_device -d /dev/xdma0_h2c_0 -f {} -a 0x0 -s {}".format(file_name, size)
    #print("Running command: {}".format(command))
    os.system(command)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python W-DDR.py <file_name> ")
        sys.exit(1)

    file_name = sys.argv[1]
    # size = sys.argv[2]
    size=57376
    run_dma_to_device(file_name, size)
