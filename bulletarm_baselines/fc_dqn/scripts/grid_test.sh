#!/bin/bash
for ((i=0;i<10;i++));do

{
    python bulletarm_baselines/fc_dqn/scripts/grid_test.py --device_name=cuda --num_process=0 --load_model_pre="/root/snapshot" --num_object=1 --object_index=${i}
    
}
done