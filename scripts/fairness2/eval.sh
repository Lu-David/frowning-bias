#!/bin/bash
python3 predict.py --model-dir "./results/fairness2" --model-file "ResNet18_lr0.001_bs32_optAdamW_wd0.001_sch_step_pp3_bp3_trtrain_files.csv_vaval_files.csv_tfnone_do0.0_cwTrue_eqnone_eqhownone_1670106602_model.pt" --model-name "fairness2" --architecture "ResNet18" --results-dir "./results/fairness2"
python3 eval.py --predictions-dir "./results/fairness2" --predictions-file "fairness2_predicttest_files.csv.npz"
