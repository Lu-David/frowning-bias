#!/bin/bash
python3 predict.py --model-dir "./results/fairness2/affirmative_action_10x" --model-file "ResNet18_lr0.001_bs32_optAdamW_wd0.001_sch_step_pp3_bp3_trtrain_files.csv_vaval_files.csv_tfnone_do0.0_cwTrue_eqnone_eqhownone_ow0_0_0_1670176377_model.pt" --model-name "fairness2-000" --architecture "ResNet18" --results-dir "./results/fairness2/affirmative_action_10x"
python3 eval.py --predictions-dir "./results/fairness2/affirmative_action_10x" --predictions-file "fairness2-000_predicttest_files.csv.npz"

python3 predict.py --model-dir "./results/fairness2/affirmative_action_10x" --model-file "ResNet18_lr0.001_bs32_optAdamW_wd0.001_sch_step_pp3_bp3_trtrain_files.csv_vaval_files.csv_tfnone_do0.0_cwTrue_eqnone_eqhownone_ow2_0_0_1670185455_model.pt" --model-name "fairness2-200" --architecture "ResNet18" --results-dir "./results/fairness2/affirmative_action_10x"
python3 eval.py --predictions-dir "./results/fairness2/affirmative_action_10x" --predictions-file "fairness2-200_predicttest_files.csv.npz"
