#!/bin/bash
# fairness2: affr action
python3 train_all_affraction.py --architecture "ResNet18" --pretrained "y" --frozen "n" --initial-lr 0.001 --optimizer-family "AdamW" --batch-size 32 --break-patience 3 --weight-decay 0.001 --dropout 0 --class-weights "y" --equalized-by "none" --equalized-how "none" --over-weight "0_0_0" --print-batches "y" --results-dir "./results/fairness2/affirmative_action_10x"
python3 train_all_affraction.py --architecture "ResNet18" --pretrained "y" --frozen "n" --initial-lr 0.001 --optimizer-family "AdamW" --batch-size 32 --break-patience 3 --weight-decay 0.001 --dropout 0 --class-weights "y" --equalized-by "none" --equalized-how "none" --over-weight "0_0_1" --print-batches "y" --results-dir "./results/fairness2/affirmative_action_10x"
python3 train_all_affraction.py --architecture "ResNet18" --pretrained "y" --frozen "n" --initial-lr 0.001 --optimizer-family "AdamW" --batch-size 32 --break-patience 3 --weight-decay 0.001 --dropout 0 --class-weights "y" --equalized-by "none" --equalized-how "none" --over-weight "0_0_2" --print-batches "y" --results-dir "./results/fairness2/affirmative_action_10x"
python3 train_all_affraction.py --architecture "ResNet18" --pretrained "y" --frozen "n" --initial-lr 0.001 --optimizer-family "AdamW" --batch-size 32 --break-patience 3 --weight-decay 0.001 --dropout 0 --class-weights "y" --equalized-by "none" --equalized-how "none" --over-weight "0_0_3" --print-batches "y" --results-dir "./results/fairness2/affirmative_action_10x"
python3 train_all_affraction.py --architecture "ResNet18" --pretrained "y" --frozen "n" --initial-lr 0.001 --optimizer-family "AdamW" --batch-size 32 --break-patience 3 --weight-decay 0.001 --dropout 0 --class-weights "y" --equalized-by "none" --equalized-how "none" --over-weight "0_0_4" --print-batches "y" --results-dir "./results/fairness2/affirmative_action_10x"
python3 train_all_affraction.py --architecture "ResNet18" --pretrained "y" --frozen "n" --initial-lr 0.001 --optimizer-family "AdamW" --batch-size 32 --break-patience 3 --weight-decay 0.001 --dropout 0 --class-weights "y" --equalized-by "none" --equalized-how "none" --over-weight "0_1_0" --print-batches "y" --results-dir "./results/fairness2/affirmative_action_10x"
python3 train_all_affraction.py --architecture "ResNet18" --pretrained "y" --frozen "n" --initial-lr 0.001 --optimizer-family "AdamW" --batch-size 32 --break-patience 3 --weight-decay 0.001 --dropout 0 --class-weights "y" --equalized-by "none" --equalized-how "none" --over-weight "0_1_1" --print-batches "y" --results-dir "./results/fairness2/affirmative_action_10x"
python3 train_all_affraction.py --architecture "ResNet18" --pretrained "y" --frozen "n" --initial-lr 0.001 --optimizer-family "AdamW" --batch-size 32 --break-patience 3 --weight-decay 0.001 --dropout 0 --class-weights "y" --equalized-by "none" --equalized-how "none" --over-weight "0_1_2" --print-batches "y" --results-dir "./results/fairness2/affirmative_action_10x"
python3 train_all_affraction.py --architecture "ResNet18" --pretrained "y" --frozen "n" --initial-lr 0.001 --optimizer-family "AdamW" --batch-size 32 --break-patience 3 --weight-decay 0.001 --dropout 0 --class-weights "y" --equalized-by "none" --equalized-how "none" --over-weight "0_1_3" --print-batches "y" --results-dir "./results/fairness2/affirmative_action_10x"
python3 train_all_affraction.py --architecture "ResNet18" --pretrained "y" --frozen "n" --initial-lr 0.001 --optimizer-family "AdamW" --batch-size 32 --break-patience 3 --weight-decay 0.001 --dropout 0 --class-weights "y" --equalized-by "none" --equalized-how "none" --over-weight "0_1_4" --print-batches "y" --results-dir "./results/fairness2/affirmative_action_10x"
python3 train_all_affraction.py --architecture "ResNet18" --pretrained "y" --frozen "n" --initial-lr 0.001 --optimizer-family "AdamW" --batch-size 32 --break-patience 3 --weight-decay 0.001 --dropout 0 --class-weights "y" --equalized-by "none" --equalized-how "none" --over-weight "0_2_0" --print-batches "y" --results-dir "./results/fairness2/affirmative_action_10x"
python3 train_all_affraction.py --architecture "ResNet18" --pretrained "y" --frozen "n" --initial-lr 0.001 --optimizer-family "AdamW" --batch-size 32 --break-patience 3 --weight-decay 0.001 --dropout 0 --class-weights "y" --equalized-by "none" --equalized-how "none" --over-weight "0_2_1" --print-batches "y" --results-dir "./results/fairness2/affirmative_action_10x"
python3 train_all_affraction.py --architecture "ResNet18" --pretrained "y" --frozen "n" --initial-lr 0.001 --optimizer-family "AdamW" --batch-size 32 --break-patience 3 --weight-decay 0.001 --dropout 0 --class-weights "y" --equalized-by "none" --equalized-how "none" --over-weight "0_2_2" --print-batches "y" --results-dir "./results/fairness2/affirmative_action_10x"
python3 train_all_affraction.py --architecture "ResNet18" --pretrained "y" --frozen "n" --initial-lr 0.001 --optimizer-family "AdamW" --batch-size 32 --break-patience 3 --weight-decay 0.001 --dropout 0 --class-weights "y" --equalized-by "none" --equalized-how "none" --over-weight "0_2_3" --print-batches "y" --results-dir "./results/fairness2/affirmative_action_10x"
python3 train_all_affraction.py --architecture "ResNet18" --pretrained "y" --frozen "n" --initial-lr 0.001 --optimizer-family "AdamW" --batch-size 32 --break-patience 3 --weight-decay 0.001 --dropout 0 --class-weights "y" --equalized-by "none" --equalized-how "none" --over-weight "0_2_4" --print-batches "y" --results-dir "./results/fairness2/affirmative_action_10x"

python3 train_all_affraction.py --architecture "ResNet18" --pretrained "y" --frozen "n" --initial-lr 0.001 --optimizer-family "AdamW" --batch-size 32 --break-patience 3 --weight-decay 0.001 --dropout 0 --class-weights "y" --equalized-by "none" --equalized-how "none" --over-weight "1_0_0" --print-batches "y" --results-dir "./results/fairness2/affirmative_action_10x"
python3 train_all_affraction.py --architecture "ResNet18" --pretrained "y" --frozen "n" --initial-lr 0.001 --optimizer-family "AdamW" --batch-size 32 --break-patience 3 --weight-decay 0.001 --dropout 0 --class-weights "y" --equalized-by "none" --equalized-how "none" --over-weight "1_0_1" --print-batches "y" --results-dir "./results/fairness2/affirmative_action_10x"
python3 train_all_affraction.py --architecture "ResNet18" --pretrained "y" --frozen "n" --initial-lr 0.001 --optimizer-family "AdamW" --batch-size 32 --break-patience 3 --weight-decay 0.001 --dropout 0 --class-weights "y" --equalized-by "none" --equalized-how "none" --over-weight "1_0_2" --print-batches "y" --results-dir "./results/fairness2/affirmative_action_10x"
python3 train_all_affraction.py --architecture "ResNet18" --pretrained "y" --frozen "n" --initial-lr 0.001 --optimizer-family "AdamW" --batch-size 32 --break-patience 3 --weight-decay 0.001 --dropout 0 --class-weights "y" --equalized-by "none" --equalized-how "none" --over-weight "1_0_3" --print-batches "y" --results-dir "./results/fairness2/affirmative_action_10x"
python3 train_all_affraction.py --architecture "ResNet18" --pretrained "y" --frozen "n" --initial-lr 0.001 --optimizer-family "AdamW" --batch-size 32 --break-patience 3 --weight-decay 0.001 --dropout 0 --class-weights "y" --equalized-by "none" --equalized-how "none" --over-weight "1_0_4" --print-batches "y" --results-dir "./results/fairness2/affirmative_action_10x"
python3 train_all_affraction.py --architecture "ResNet18" --pretrained "y" --frozen "n" --initial-lr 0.001 --optimizer-family "AdamW" --batch-size 32 --break-patience 3 --weight-decay 0.001 --dropout 0 --class-weights "y" --equalized-by "none" --equalized-how "none" --over-weight "1_1_0" --print-batches "y" --results-dir "./results/fairness2/affirmative_action_10x"
python3 train_all_affraction.py --architecture "ResNet18" --pretrained "y" --frozen "n" --initial-lr 0.001 --optimizer-family "AdamW" --batch-size 32 --break-patience 3 --weight-decay 0.001 --dropout 0 --class-weights "y" --equalized-by "none" --equalized-how "none" --over-weight "1_1_1" --print-batches "y" --results-dir "./results/fairness2/affirmative_action_10x"
python3 train_all_affraction.py --architecture "ResNet18" --pretrained "y" --frozen "n" --initial-lr 0.001 --optimizer-family "AdamW" --batch-size 32 --break-patience 3 --weight-decay 0.001 --dropout 0 --class-weights "y" --equalized-by "none" --equalized-how "none" --over-weight "1_1_2" --print-batches "y" --results-dir "./results/fairness2/affirmative_action_10x"
python3 train_all_affraction.py --architecture "ResNet18" --pretrained "y" --frozen "n" --initial-lr 0.001 --optimizer-family "AdamW" --batch-size 32 --break-patience 3 --weight-decay 0.001 --dropout 0 --class-weights "y" --equalized-by "none" --equalized-how "none" --over-weight "1_1_3" --print-batches "y" --results-dir "./results/fairness2/affirmative_action_10x"
python3 train_all_affraction.py --architecture "ResNet18" --pretrained "y" --frozen "n" --initial-lr 0.001 --optimizer-family "AdamW" --batch-size 32 --break-patience 3 --weight-decay 0.001 --dropout 0 --class-weights "y" --equalized-by "none" --equalized-how "none" --over-weight "1_1_4" --print-batches "y" --results-dir "./results/fairness2/affirmative_action_10x"
python3 train_all_affraction.py --architecture "ResNet18" --pretrained "y" --frozen "n" --initial-lr 0.001 --optimizer-family "AdamW" --batch-size 32 --break-patience 3 --weight-decay 0.001 --dropout 0 --class-weights "y" --equalized-by "none" --equalized-how "none" --over-weight "1_2_0" --print-batches "y" --results-dir "./results/fairness2/affirmative_action_10x"
python3 train_all_affraction.py --architecture "ResNet18" --pretrained "y" --frozen "n" --initial-lr 0.001 --optimizer-family "AdamW" --batch-size 32 --break-patience 3 --weight-decay 0.001 --dropout 0 --class-weights "y" --equalized-by "none" --equalized-how "none" --over-weight "1_2_1" --print-batches "y" --results-dir "./results/fairness2/affirmative_action_10x"
python3 train_all_affraction.py --architecture "ResNet18" --pretrained "y" --frozen "n" --initial-lr 0.001 --optimizer-family "AdamW" --batch-size 32 --break-patience 3 --weight-decay 0.001 --dropout 0 --class-weights "y" --equalized-by "none" --equalized-how "none" --over-weight "1_2_2" --print-batches "y" --results-dir "./results/fairness2/affirmative_action_10x"
python3 train_all_affraction.py --architecture "ResNet18" --pretrained "y" --frozen "n" --initial-lr 0.001 --optimizer-family "AdamW" --batch-size 32 --break-patience 3 --weight-decay 0.001 --dropout 0 --class-weights "y" --equalized-by "none" --equalized-how "none" --over-weight "1_2_3" --print-batches "y" --results-dir "./results/fairness2/affirmative_action_10x"
python3 train_all_affraction.py --architecture "ResNet18" --pretrained "y" --frozen "n" --initial-lr 0.001 --optimizer-family "AdamW" --batch-size 32 --break-patience 3 --weight-decay 0.001 --dropout 0 --class-weights "y" --equalized-by "none" --equalized-how "none" --over-weight "1_2_4" --print-batches "y" --results-dir "./results/fairness2/affirmative_action_10x"

python3 train_all_affraction.py --architecture "ResNet18" --pretrained "y" --frozen "n" --initial-lr 0.001 --optimizer-family "AdamW" --batch-size 32 --break-patience 3 --weight-decay 0.001 --dropout 0 --class-weights "y" --equalized-by "none" --equalized-how "none" --over-weight "2_0_0" --print-batches "y" --results-dir "./results/fairness2/affirmative_action_10x"
python3 train_all_affraction.py --architecture "ResNet18" --pretrained "y" --frozen "n" --initial-lr 0.001 --optimizer-family "AdamW" --batch-size 32 --break-patience 3 --weight-decay 0.001 --dropout 0 --class-weights "y" --equalized-by "none" --equalized-how "none" --over-weight "2_0_1" --print-batches "y" --results-dir "./results/fairness2/affirmative_action_10x"
python3 train_all_affraction.py --architecture "ResNet18" --pretrained "y" --frozen "n" --initial-lr 0.001 --optimizer-family "AdamW" --batch-size 32 --break-patience 3 --weight-decay 0.001 --dropout 0 --class-weights "y" --equalized-by "none" --equalized-how "none" --over-weight "2_0_2" --print-batches "y" --results-dir "./results/fairness2/affirmative_action_10x"
python3 train_all_affraction.py --architecture "ResNet18" --pretrained "y" --frozen "n" --initial-lr 0.001 --optimizer-family "AdamW" --batch-size 32 --break-patience 3 --weight-decay 0.001 --dropout 0 --class-weights "y" --equalized-by "none" --equalized-how "none" --over-weight "2_0_3" --print-batches "y" --results-dir "./results/fairness2/affirmative_action_10x"
python3 train_all_affraction.py --architecture "ResNet18" --pretrained "y" --frozen "n" --initial-lr 0.001 --optimizer-family "AdamW" --batch-size 32 --break-patience 3 --weight-decay 0.001 --dropout 0 --class-weights "y" --equalized-by "none" --equalized-how "none" --over-weight "2_0_4" --print-batches "y" --results-dir "./results/fairness2/affirmative_action_10x"
python3 train_all_affraction.py --architecture "ResNet18" --pretrained "y" --frozen "n" --initial-lr 0.001 --optimizer-family "AdamW" --batch-size 32 --break-patience 3 --weight-decay 0.001 --dropout 0 --class-weights "y" --equalized-by "none" --equalized-how "none" --over-weight "2_1_0" --print-batches "y" --results-dir "./results/fairness2/affirmative_action_10x"
python3 train_all_affraction.py --architecture "ResNet18" --pretrained "y" --frozen "n" --initial-lr 0.001 --optimizer-family "AdamW" --batch-size 32 --break-patience 3 --weight-decay 0.001 --dropout 0 --class-weights "y" --equalized-by "none" --equalized-how "none" --over-weight "2_1_1" --print-batches "y" --results-dir "./results/fairness2/affirmative_action_10x"
python3 train_all_affraction.py --architecture "ResNet18" --pretrained "y" --frozen "n" --initial-lr 0.001 --optimizer-family "AdamW" --batch-size 32 --break-patience 3 --weight-decay 0.001 --dropout 0 --class-weights "y" --equalized-by "none" --equalized-how "none" --over-weight "2_1_2" --print-batches "y" --results-dir "./results/fairness2/affirmative_action_10x"
python3 train_all_affraction.py --architecture "ResNet18" --pretrained "y" --frozen "n" --initial-lr 0.001 --optimizer-family "AdamW" --batch-size 32 --break-patience 3 --weight-decay 0.001 --dropout 0 --class-weights "y" --equalized-by "none" --equalized-how "none" --over-weight "2_1_3" --print-batches "y" --results-dir "./results/fairness2/affirmative_action_10x"
python3 train_all_affraction.py --architecture "ResNet18" --pretrained "y" --frozen "n" --initial-lr 0.001 --optimizer-family "AdamW" --batch-size 32 --break-patience 3 --weight-decay 0.001 --dropout 0 --class-weights "y" --equalized-by "none" --equalized-how "none" --over-weight "2_1_4" --print-batches "y" --results-dir "./results/fairness2/affirmative_action_10x"
python3 train_all_affraction.py --architecture "ResNet18" --pretrained "y" --frozen "n" --initial-lr 0.001 --optimizer-family "AdamW" --batch-size 32 --break-patience 3 --weight-decay 0.001 --dropout 0 --class-weights "y" --equalized-by "none" --equalized-how "none" --over-weight "2_2_0" --print-batches "y" --results-dir "./results/fairness2/affirmative_action_10x"
python3 train_all_affraction.py --architecture "ResNet18" --pretrained "y" --frozen "n" --initial-lr 0.001 --optimizer-family "AdamW" --batch-size 32 --break-patience 3 --weight-decay 0.001 --dropout 0 --class-weights "y" --equalized-by "none" --equalized-how "none" --over-weight "2_2_1" --print-batches "y" --results-dir "./results/fairness2/affirmative_action_10x"
python3 train_all_affraction.py --architecture "ResNet18" --pretrained "y" --frozen "n" --initial-lr 0.001 --optimizer-family "AdamW" --batch-size 32 --break-patience 3 --weight-decay 0.001 --dropout 0 --class-weights "y" --equalized-by "none" --equalized-how "none" --over-weight "2_2_2" --print-batches "y" --results-dir "./results/fairness2/affirmative_action_10x"
python3 train_all_affraction.py --architecture "ResNet18" --pretrained "y" --frozen "n" --initial-lr 0.001 --optimizer-family "AdamW" --batch-size 32 --break-patience 3 --weight-decay 0.001 --dropout 0 --class-weights "y" --equalized-by "none" --equalized-how "none" --over-weight "2_2_3" --print-batches "y" --results-dir "./results/fairness2/affirmative_action_10x"
python3 train_all_affraction.py --architecture "ResNet18" --pretrained "y" --frozen "n" --initial-lr 0.001 --optimizer-family "AdamW" --batch-size 32 --break-patience 3 --weight-decay 0.001 --dropout 0 --class-weights "y" --equalized-by "none" --equalized-how "none" --over-weight "2_2_4" --print-batches "y" --results-dir "./results/fairness2/affirmative_action_10x"

