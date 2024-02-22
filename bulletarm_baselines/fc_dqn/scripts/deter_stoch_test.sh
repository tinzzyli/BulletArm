#!/bin/bash
arr=(84 82 81 79 77 74 73 67 56 51 48 47 33 25 20 19 18 9 6 5 3 2 1 0)
# for ((i=3;i>=0;i--));do
for i in ${arr[@]}; do
{
    python bulletarm_baselines/fc_dqn/scripts/deter_stoch_test.py --device_name=cuda --num_process=0 --load_model_pre="/root/snapshot" --num_object=1 --object_index=${i}
    
}
done