#!/usr/bin/env bash
python -u -m imagenet_ddp_RandConv_3 --max_epoch 10  --optimizer Adam --network DeiTSmall  --log_path Results/RandConv_DeiTSmall_with_ranConv_validation  --wd 0 --world_size 3 --batch_size 64