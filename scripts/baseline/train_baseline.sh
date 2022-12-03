#!/bin/bash
python3 train_all.py --architecture "ResNet18" --pretrained "y" --frozen "n" --initial-lr 0.001 --optimizer-family "AdamW" --batch-size 32 --break-patience 3 --weight-decay 0.001 --dropout 0 --class-weights "y" --print-batches "y" --results-dir "./results/baseline"
