#!/usr/bin/env bash

DATASETS=("domainD-V2" "domainD-S" "domainD-R" "domainD-Cue" "domainD-C" "domainD-9")

for DATASET in "${DATASETS[@]}"; do
    echo "Running domain_distinguishability for dataset: $DATASET"
    python -u -m domain_distinguishability --baseline --dataset "$DATASET" --seed 0 --network ViTBase --algorithm RandConv --log_path Results/RandConv_ViTBase_Head_ReInitialized_with_ranConv_validation
done


# python -u -m binary_domain_distinguishability --baseline --dataset domainD-9V2 --seed 0 --network ResNet50  --algorithm RandConv --log_path Results/RandConv_CNN_Head_ReInitialized_with_ranConv_validation_different_ERM