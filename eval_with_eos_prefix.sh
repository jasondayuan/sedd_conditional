#!/bin/bash

datasets=("lambada")
mask_probs=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)
mean_span_lengths=(3)

for dataset in "${datasets[@]}"; do
    for mask_prob in "${mask_probs[@]}"; do
        for mean_span_length in "${mean_span_lengths[@]}"; do
            while true; do
                python eval_conditional_ppl.py \
                --model_size "small" \
                --dataset_name $dataset \
                --batch_size 16 \
                --mask_prob $mask_prob \
                --mean_span_length $mean_span_length \
                --mask_type "prefix" \
                --include_eos \
                --block_size 128
                if [ $? -eq 0 ]; then
                    break
                else
                    echo "Script failed for dataset $dataset, mask_prob $mask_prob, mask_type $mask_type. Retrying..."
                fi
            done
        done
    done
done