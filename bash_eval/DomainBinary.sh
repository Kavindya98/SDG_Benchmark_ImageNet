#!/usr/bin/env bash

#python -u -m binary_domain_distinguishability --lr 0.005 --wd 0.8  --dataset domainD-R_V2   --seed 0 --network ViTBase  --algorithm RandConv --log_path Results/RandConv_ViTBase_Head_ReInitialized_with_ranConv_validation
#python -u -m binary_domain_distinguishability --lr 0.0005 --wd 0.8  --dataset domainD-V2  --seed 0 --network ViTBase  --algorithm RandConv --log_path Results/RandConv_ViTBase_Head_ReInitialized_with_ranConv_validation
# python -u -m binary_domain_distinguishability --lr 0.0007 --wd 0.5  --dataset domainD-C  --seed 0 --network ViTBase  --algorithm RandConv --log_path Results/RandConv_ViTBase_Head_ReInitialized_with_ranConv_validation


#python -u -m binary_domain_distinguishability --lr 0.001 --wd 0.5  --dataset domainD-S  --seed 0 --network ViTBase  --algorithm RandConv --log_path Results/RandConv_ViTBase_Head_ReInitialized_with_ranConv_validation
python -u -m binary_domain_distinguishability --lr 0.001 --wd 0.5  --dataset domainD-CueV4  --seed 0 --network ViTBase  --algorithm RandConv --log_path Results/RandConv_ViTBase_Head_ReInitialized_with_ranConv_validation