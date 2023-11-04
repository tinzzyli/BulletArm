#!/bin/bash

values=("gaussian_noise" "shot_noise" "impulse_noise" "motion_blur" "zoom_blur" "fog" "brightness" "contrast" "jpeg_compression")

for value in "${values[@]}";do
{
    !python ./common_corruptions/corruption_test.py  --device_name=cuda --env=house_building_3 --corrupt_func="$value" --severity=5 --load_model_pre=/content/drive/MyDrive/my_archive/house_building_3/snapshot
}
done