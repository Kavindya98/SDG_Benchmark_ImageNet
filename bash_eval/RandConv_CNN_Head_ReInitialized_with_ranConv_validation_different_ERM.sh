#!/usr/bin/env bash
python -u -m evaluvation --loss_aug --max_epoch 30 --Freeze_bn --optimizer SGD --network ResNet50  --log_path Results/RandConv_CNN_Head_ReInitialized_with_ranConv_validation_different_ERM --lr 0.0001 --wd 0 --world_size 4 --batch_size 64 --pretrained --pretrained_custom_weights --head_re_initialized