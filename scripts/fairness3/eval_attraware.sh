#!/bin/bash
python3 predict_attraware.py --model-dir "./results/fairness3/" --model-file "ResNet18_lr0.001_bs32_optAdamW_wd0.001_sch_step_pp3_bp3_trtrain_files.csv_vaval_files.csv_tfnone_do0.0_cwTrue_eqnone_eqhownone_1670310188_model.pt" --model-name "fairness3-attraware" --architecture "ResNet18" --results-dir "./results/fairness3/"
python3 eval.py --predictions-dir "./results/fairness3/" --predictions-file "fairness3-attraware_predicttest_files.csv.npz"
