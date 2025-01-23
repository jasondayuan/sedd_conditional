#!/bin/bash

datasets=("wikitext" "1bw")

for dataset in "${datasets[@]}"; do
    while true; do
        python eval_conditional_ppl.py \
        --model_size "small" \
        --dataset_name $dataset \
        --mask_type "front_half" \
        --batch_size 4 \
        --include_eos
        if [ $? -eq 0 ]; then
            break
        else
            echo "Script failed for dataset $dataset, mask_prob $mask_prob, mask_type $mask_type. Retrying..."
        fi
    done
done