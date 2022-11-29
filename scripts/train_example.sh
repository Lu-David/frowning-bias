#!/bin/bash
python3 train.py --architecture "ResNet18" --pretrained "y" --frozen "n" --batch-size 32 --dropout 0.5 --print-batches "y" --results-dir "./results/scratch"
