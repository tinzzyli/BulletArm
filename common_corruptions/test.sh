#!/bin/bash

values=("gaussian_noise" "poisson_noise" "salt_pepper_noise", "rotation", "translation")

for value in "${values[@]}";do
{
    python ./common_corruptions/corruption_test.py  --device_name=cuda --env=house_building_3 --corrupt_func="$value" --severity=5 --load_model_pre=/content/drive/MyDrive/my_archive/house_building_3/snapshot
    python ./common_corruptions/corruption_test.py  --device_name=cuda --env=house_building_3 --corrupt_func="$value" --severity=4 --load_model_pre=/content/drive/MyDrive/my_archive/house_building_3/snapshot
}
done