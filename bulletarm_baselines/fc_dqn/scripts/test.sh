#!/bin/bash

for ((i=20;i<40;i++));do

{
    python bulletarm_baselines/fc_dqn/scripts/pgd_attack.py --device_name=cuda --num_process=0 --load_model_pre="/content/drive/MyDrive/my_archive/file/ck5/snapshot" --num_object=1 --object_index=${i}
    
}
done