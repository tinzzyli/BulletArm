#!/bin/bash

for ((i=0;i<5;i++));do

{
    python bulletarm_baselines/fc_dqn/scripts/pgd_attack.py --device_name=cuda --num_process=0 --load_model_pre="/root/snapshot" --num_object=1 --object_index=${i}
    
}
done