#!/usr/bin/env bash
python -u -m imagenet_ddp_ABA  --max_epoch 30 --lr_adv 0.00005 --Freeze_bn --optimizer SGD --elbo_bet 0.1 --lr_scheduler CosineLR --num_blocks 0 --pre_epoch 4 --network ResNet50  --log_path Results/ABA_CNN_Head_ReInitialized_different_ERM_3 --lr 0.004 --wd 0 --world_size 4 --adv_steps 10 --batch_size 64 --pretrained --pretrained_custom_weights --head_re_initialized

# CUDA_VISIBLE_DEVICES=2,3 nohup bash bash/run_ABA.sh > nohup/ABA_CNN_Head_ReInitialized_different_ERM_3.out 2>&1 &