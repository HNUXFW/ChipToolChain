#!/bin/sh
exec 2>/dev/null
# pciew #############
# xdma device
#DEVICE_PATH= /dev/xdma0_user

# Assign Write addr-data
# addr = 4096 + (32)
# num =  1
# start = 0x8000_0000
WRITE_ADDR_ARRAY=(0x00   0x00   0x04   0x04   0x08   0x08   0x0c   0x0c)
WRITE_DATA_ARRAY=(0x0000 0x0000 0x000a 0x0000 0x0000 0x8000 0x6666 0x5555)
#Multi-Write
for ((i = 0 ; i <  ${#WRITE_ADDR_ARRAY[@]}; i++ )); do
	WRITE_ADDR=${WRITE_ADDR_ARRAY[i]}
     	WRITE_DATA=${WRITE_DATA_ARRAY[i]}
       	./reg_rw /dev/xdma0_user $WRITE_ADDR w $WRITE_DATA
	echo 
	echo 1#######################################
        echo 1#### A write transfer is completed#####
	echo 1#######################################
	echo 
	#sleep 3.6
done

echo 
echo Write Data complete!
echo
