#!/usr/bin/env bash
python -u -m evaluvation --max_epoch 30 --optimizer Adam --network ViTBase --log_path Results/ViTBase_AugMix_CNN_Head_ReInitialized_2 --wd 0 --world_size 4 --batch_size 64 --pretrained --head_re_initialized