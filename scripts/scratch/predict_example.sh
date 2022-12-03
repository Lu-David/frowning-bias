#!/bin/bash
# python3 predict.py --model-dir "./results/hparam" --model-file "ResNet18_lr0.001_bs32_optAdamW_wd0.001_sch_step_pp3_bp3_trtrain_files.csv_vaval_files.csv_tfnone_do0.0_1669721551_model.pt" --model-name "hparam-auc-1" --architecture "ResNet18" --results-dir "./results/scratch"
# python3 predict.py --model-dir "./results/scratch" --model-file "ResNet18_lr0.001_bs32_optAdamW_wd0.001_sch_step_pp3_bp3_trtrain_files.csv_vaval_files.csv_tfnone_do0.0_swTrue_1669831367_model.pt" --model-name "hparam-sw-auc-1" --architecture "ResNet18" --results-dir "./results/scratch"
python3 predict.py --model-dir "./results/scratch" --model-file "ResNet18_lr0.001_bs32_optAdamW_wd0.001_sch_step_pp3_bp3_trtrain_files.csv_vaval_files.csv_tfnone_do0.0_swTrue_1669933781_FAIR_model.pt" --model-name "hparam-sw-fair1-auc-1" --architecture "ResNet18" --results-dir "./results/scratch"


# results/scratch/ResNet18_lr0.001_bs32_optAdamW_wd0.001_sch_step_pp3_bp3_trtrain_files.csv_vaval_files.csv_tfnone_do0.0_swTrue_1669831367_model.pt
# /home/kesavan/Documents/cs475/FrowningUponBias/results/scratch/ResNet18_lr0.001_bs32_optAdamW_wd0.001_sch_step_pp3_bp3_trtrain_files.csv_vaval_files.csv_tfnone_do0.0_swTrue_1669933781_FAIR_model.pt