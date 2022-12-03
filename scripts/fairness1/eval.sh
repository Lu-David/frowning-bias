#!/bin/bash
python3 predict.py --model-dir "./results/fairness1" --model-file "ResNet18_lr0.001_bs32_optAdamW_wd0.001_sch_step_pp3_bp3_trtrain_files.csv_vaval_files.csv_tfnone_do0.0_swTrue_eqAge_eqhowdown_1670093528_model.pt" --model-name "fairness1" --architecture "ResNet18" --results-dir "./results/fairness1"
python3 eval.py --predictions-dir "./results/fairness1" --predictions-file "fairness1_predicttest_files.csv.npz"
