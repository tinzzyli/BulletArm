#!/bin/bash

# for ((j=-4;j>=-7;j--));do
# {
#     CUDA_VISIBLE_DEVICES=1 python main.py --env=object_grasping --object_index=${j}
#     mv ../../../pos_info.txt ./pos_info_${j}.txt
# }
# done

python main.py --env=object_grasping --object_index=1