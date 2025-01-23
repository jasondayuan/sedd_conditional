#!/bin/bash

datasets=("wikitext" "1bw")
mask_probs=(0.1 0.3 0.5 0.7 0.9)
mean_span_lengths=(3 50)

for dataset in "${datasets[@]}"; do
    for mask_prob in "${mask_probs[@]}"; do
        for mean_span_length in "${mean_span_lengths[@]}"; do
            while true; do
                python eval_conditional_ppl.py \
                --model_size "medium" \
                --dataset_name $dataset \
                --mask_prob $mask_prob \
                --mean_span_length $mean_span_length \
                --batch_size 8
                if [ $? -eq 0 ]; then
                    break
                else
                    echo "Script failed for dataset $dataset, mask_prob $mask_prob, mask_type $mask_type. Retrying..."
                fi
            done
        done
    done
done