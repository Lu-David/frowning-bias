#!/bin/bash
# python3 train.py --architecture "ResNet18" --pretrained "y" --frozen "n" --batch-size 32 --dropout 0.5 --print-batches "y" --results-dir "./results/scratch"
# python3 train.py --architecture "ResNet18" --pretrained "y" --frozen "n" --initial-lr 0.001 --optimizer-family "AdamW" --batch-size 32 --break-patience 3 --weight-decay 0.001 --dropout 0 --sample-weights "y" --print-batches "y" --results-dir "./results/scratch"
# python3 train_withfairness.py --architecture "ResNet18" --pretrained "y" --frozen "n" --initial-lr 0.001 --optimizer-family "AdamW" --batch-size 32 --break-patience 3 --weight-decay 0.001 --dropout 0 --sample-weights "y" --print-batches "y" --results-dir "./results/scratch"


python3 train_all.py --architecture "ResNet18" --pretrained "y" --frozen "n" --initial-lr 0.001 --optimizer-family "AdamW" --batch-size 32 --break-patience 3 --weight-decay 0.001 --dropout 0 --sample-weights "y" --equalized-by "Age" --print-batches "y" --results-dir "./results/fairness1"
