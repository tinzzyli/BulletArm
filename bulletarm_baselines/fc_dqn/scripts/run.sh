#!/bin/bash

CUDA_VISIBLE_DEVICES=1 python test.py --load_model_pre='/home/mxh170530/BulletArm/bulletarm_baselines/fc_dqn/scripts/output/margin_asr_equ_resu_df_flip_object_grasping/2022-10-10.20:06:28/models/snapshot' --env=object_grasping --object_index=0

# for ((i=1;i<16;i++));do
# {
#     CUDA_VISIBLE_DEVICES=1 python test.py --load_model_pre='/home/zoooe/workspaces/BulletArm/bulletarm_baselines/fc_dqn/scripts/output/margin_asr_equ_resu_df_flip_object_grasping/2022-09-19.18:34:35/models/snapshot' --env=object_grasping --num_objects=${i}
# }
# done

# for ((j=0;j<8;j++));do
# {
#     CUDA_VISIBLE_DEVICES=1 python test.py --load_model_pre='/home/mxh170530/BulletArm/bulletarm_baselines/fc_dqn/scripts/output/margin_asr_equ_resu_df_flip_object_grasping/2022-10-10.20:06:28/models/snapshot' --env=object_grasping --object_index=${j}
# }
# done


# for ((j=8;j<17;j++));do
# {
#     CUDA_VISIBLE_DEVICES=1 python test.py --load_model_pre='/home/mxh170530/BulletArm/bulletarm_baselines/fc_dqn/scripts/output/margin_asr_equ_resu_df_flip_object_grasping/2022-10-11.07:22:47/models/snapshot' --env=object_grasping --object_index=${j}
# }
# done

# for ((j=17;j<25;j++));do
# {
#     CUDA_VISIBLE_DEVICES=1 python test.py --load_model_pre='/home/mxh170530/BulletArm/bulletarm_baselines/fc_dqn/scripts/output/margin_asr_equ_resu_df_flip_object_grasping/2022-10-11.18:53:52/models/snapshot' --env=object_grasping --object_index=${j}
# }
# done


# for ((j=25;j<34;j++));do
# {
#     CUDA_VISIBLE_DEVICES=1 python test.py --load_model_pre='/home/mxh170530/BulletArm/bulletarm_baselines/fc_dqn/scripts/output/margin_asr_equ_resu_df_flip_object_grasping/2022-10-19.11:28:15/models/snapshot' --env=object_grasping --object_index=${j}
# }
# done


# for ((j=34;j<43;j++));do
# {
#     CUDA_VISIBLE_DEVICES=1 python test.py --load_model_pre='/home/mxh170530/BulletArm/bulletarm_baselines/fc_dqn/scripts/output/margin_asr_equ_resu_df_flip_object_grasping/2022-10-19.23:46:33/models/snapshot' --env=object_grasping --object_index=${j}
# }
# done


# for ((j=43;j<51;j++));do
# {
#     CUDA_VISIBLE_DEVICES=1 python test.py --load_model_pre='/home/mxh170530/BulletArm/bulletarm_baselines/fc_dqn/scripts/output/margin_asr_equ_resu_df_flip_object_grasping/2022-10-20.11:45:55/models/snapshot' --env=object_grasping --object_index=${j}
# }
# done


# for ((j=51;j<60;j++));do
# {
#     CUDA_VISIBLE_DEVICES=1 python test.py --load_model_pre='/home/mxh170530/BulletArm/bulletarm_baselines/fc_dqn/scripts/output/margin_asr_equ_resu_df_flip_object_grasping/2022-10-20.23:45:25/models/snapshot' --env=object_grasping --object_index=${j}
# }
# done


# for ((j=60;j<68;j++));do
# {
#     CUDA_VISIBLE_DEVICES=1 python test.py --load_model_pre='/nas1-nfs1/home/zxs180005/BulletArm/bulletarm_baselines/fc_dqn/scripts/output/margin_asr_equ_resu_df_flip_object_grasping/2022-10-19.11:15:33/models/snapshot' --env=object_grasping --object_index=${j}
# }
# done

# for ((j=68;j<77;j++));do
# {
#     CUDA_VISIBLE_DEVICES=1 python test.py --load_model_pre='/nas1-nfs1/home/zxs180005/BulletArm/bulletarm_baselines/fc_dqn/scripts/output/margin_asr_equ_resu_df_flip_object_grasping/2022-10-19.23:37:13/models/snapshot' --env=object_grasping --object_index=${j}
# }
# done

# for ((j=77;j<86;j++));do
# {
#     CUDA_VISIBLE_DEVICES=1 python test.py --load_model_pre='/nas1-nfs1/home/zxs180005/BulletArm/bulletarm_baselines/fc_dqn/scripts/output/margin_asr_equ_resu_df_flip_object_grasping/2022-10-20.11:27:17/models/snapshot' --env=object_grasping --object_index=${j}
# }
# done
