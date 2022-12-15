#!/bin/bash
# african-american males 40-69 [103]
python3 predict.py --model-dir "./results/fairness2/affirmative_action_10x" --model-file "ResNet18_lr0.001_bs32_optAdamW_wd0.001_sch_step_pp3_bp3_trtrain_files.csv_vaval_files.csv_tfnone_do0.0_cwTrue_eqnone_eqhownone_ow1_0_3_1670182265_model.pt" --model-name "fairness2-103" --architecture "ResNet18" --results-dir "./results/fairness2/affirmative_action_10x"
python3 eval.py --predictions-dir "./results/fairness2/affirmative_action_10x" --predictions-file "fairness2-103_predicttest_files.csv.npz"
# asian females 40-69 [213]
python3 predict.py --model-dir "./results/fairness2/affirmative_action_10x" --model-file "ResNet18_lr0.001_bs32_optAdamW_wd0.001_sch_step_pp3_bp3_trtrain_files.csv_vaval_files.csv_tfnone_do0.0_cwTrue_eqnone_eqhownone_ow2_1_3_1670188086_model.pt" --model-name "fairness2-213" --architecture "ResNet18" --results-dir "./results/fairness2/affirmative_action_10x"
python3 eval.py --predictions-dir "./results/fairness2/affirmative_action_10x" --predictions-file "fairness2-213_predicttest_files.csv.npz"
