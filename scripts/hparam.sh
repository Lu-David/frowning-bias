#!/bin/bash
# Hyperparameter Search


for lr in {0.0001,0.001,0.01}
do
    for wd in {0.0001,0.001}
    do
        for d in {0,0.5}
        do
            for ep in {3,6}
            do
                for bs in {16,32}
                do
                    python3 train.py --initial-lr $lr --weight-decay $wd --break-patience $ep --batch-size $bs --train-file "train_files.csv" --val-file "val_files.csv" --dropout $d --results-dir "./results/hparam"
                done
            done
        done
    done
done
