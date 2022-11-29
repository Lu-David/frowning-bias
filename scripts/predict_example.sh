#!/bin/bash
python3 predict.py --model-dir "./results/scratch" --model-file "ResNet18_lr0.01_bs16_optAdamW_wd0.0001_sch_step_pp3_bp5_trtrain_files.csv_vaval_files.csv_tfnone_do0.5_1669689475_model.pt" --architecture "ResNet18" --results-dir "./results/scratch"
