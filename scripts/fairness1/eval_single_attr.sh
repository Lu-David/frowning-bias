#!/bin/bash
# python3 predict.py --model-dir "./results/fairness1/single_attr" --model-file "ResNet18_lr0.001_bs32_optAdamW_wd0.001_sch_step_pp3_bp3_trtrain_files.csv_vaval_files.csv_tfnone_do0.0_cwTrue_eqRace_eqhowup_1670171634_model.pt" --model-name "fairness1-race-up" --architecture "ResNet18" --results-dir "./results/fairness1/single_attr"
python3 eval.py --predictions-dir "./results/fairness1/single_attr" --predictions-file "fairness1-race-up_predicttest_files.csv.npz"

# python3 predict.py --model-dir "./results/fairness1/single_attr" --model-file "ResNet18_lr0.001_bs32_optAdamW_wd0.001_sch_step_pp3_bp3_trtrain_files.csv_vaval_files.csv_tfnone_do0.0_cwTrue_eqRace_eqhowdown_1670172319_model.pt" --model-name "fairness1-race-down" --architecture "ResNet18" --results-dir "./results/fairness1/single_attr"
python3 eval.py --predictions-dir "./results/fairness1/single_attr" --predictions-file "fairness1-race-down_predicttest_files.csv.npz"

# python3 predict.py --model-dir "./results/fairness1/single_attr" --model-file "ResNet18_lr0.001_bs32_optAdamW_wd0.001_sch_step_pp3_bp3_trtrain_files.csv_vaval_files.csv_tfnone_do0.0_cwTrue_eqRace_eqhowmean_1670172488_model.pt" --model-name "fairness1-race-mean" --architecture "ResNet18" --results-dir "./results/fairness1/single_attr"
python3 eval.py --predictions-dir "./results/fairness1/single_attr" --predictions-file "fairness1-race-mean_predicttest_files.csv.npz"

# python3 predict.py --model-dir "./results/fairness1/single_attr" --model-file "ResNet18_lr0.001_bs32_optAdamW_wd0.001_sch_step_pp3_bp3_trtrain_files.csv_vaval_files.csv_tfnone_do0.0_cwTrue_eqAge_eqhowup_1670172845_model.pt" --model-name "fairness1-age-up" --architecture "ResNet18" --results-dir "./results/fairness1/single_attr"
python3 eval.py --predictions-dir "./results/fairness1/single_attr" --predictions-file "fairness1-age-up_predicttest_files.csv.npz"

# python3 predict.py --model-dir "./results/fairness1/single_attr" --model-file "ResNet18_lr0.001_bs32_optAdamW_wd0.001_sch_step_pp3_bp3_trtrain_files.csv_vaval_files.csv_tfnone_do0.0_cwTrue_eqAge_eqhowdown_1670173378_model.pt" --model-name "fairness1-age-down" --architecture "ResNet18" --results-dir "./results/fairness1/single_attr"
python3 eval.py --predictions-dir "./results/fairness1/single_attr" --predictions-file "fairness1-age-down_predicttest_files.csv.npz"

# python3 predict.py --model-dir "./results/fairness1/single_attr" --model-file "ResNet18_lr0.001_bs32_optAdamW_wd0.001_sch_step_pp3_bp3_trtrain_files.csv_vaval_files.csv_tfnone_do0.0_cwTrue_eqAge_eqhowmean_1670173468_model.pt" --model-name "fairness1-age-mean" --architecture "ResNet18" --results-dir "./results/fairness1/single_attr"
python3 eval.py --predictions-dir "./results/fairness1/single_attr" --predictions-file "fairness1-age-mean_predicttest_files.csv.npz"

# python3 predict.py --model-dir "./results/fairness1/single_attr" --model-file "ResNet18_lr0.001_bs32_optAdamW_wd0.001_sch_step_pp3_bp3_trtrain_files.csv_vaval_files.csv_tfnone_do0.0_cwTrue_eqGender_eqhowup_1670173689_model.pt" --model-name "fairness1-gender-up" --architecture "ResNet18" --results-dir "./results/fairness1/single_attr"
python3 eval.py --predictions-dir "./results/fairness1/single_attr" --predictions-file "fairness1-gender-up_predicttest_files.csv.npz"

# python3 predict.py --model-dir "./results/fairness1/single_attr" --model-file "ResNet18_lr0.001_bs32_optAdamW_wd0.001_sch_step_pp3_bp3_trtrain_files.csv_vaval_files.csv_tfnone_do0.0_cwTrue_eqGender_eqhowdown_1670174032_model.pt" --model-name "fairness1-gender-down" --architecture "ResNet18" --results-dir "./results/fairness1/single_attr"
python3 eval.py --predictions-dir "./results/fairness1/single_attr" --predictions-file "fairness1-gender-down_predicttest_files.csv.npz"

# python3 predict.py --model-dir "./results/fairness1/single_attr" --model-file "ResNet18_lr0.001_bs32_optAdamW_wd0.001_sch_step_pp3_bp3_trtrain_files.csv_vaval_files.csv_tfnone_do0.0_cwTrue_eqGender_eqhowmean_1670174121_model.pt" --model-name "fairness1-gender-mean" --architecture "ResNet18" --results-dir "./results/fairness1/single_attr"
python3 eval.py --predictions-dir "./results/fairness1/single_attr" --predictions-file "fairness1-gender-mean_predicttest_files.csv.npz"
