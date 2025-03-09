#!/usr/bin/env bash
python -u -m imagenet_ddp_ABA  --max_epoch 30 --lr_adv 0.0005 --optimizer Adam --elbo_bet 1 --num_blocks 0 --pre_epoch 4 --network DeiTBase --log_path Results/ABA_DeiTBase_Head_ReInitialized --wd 0 --world_size 4 --adv_steps 10 --batch_size 64 --pretrained --head_re_initialized

# nohup bash bash/run_ABA_ViT.sh > nohup/ABA_ViTBase_Head_ReInitialized.out 2>&1 &