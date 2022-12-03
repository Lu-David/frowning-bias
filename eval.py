"""
Evaluate a model's predictions.
"""


# Imports
import os
import numpy as np
import pandas as pd
import time
import argparse
import warnings
import torch as T
warnings.filterwarnings("ignore", category=UserWarning)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, f1_score
import custom_modules as RAF

# Parse args
parser = argparse.ArgumentParser()
parser.add_argument("--predictions-dir", default='./results/scratch', type=str, help='directory where predictions are stored')
parser.add_argument("--predictions-file", type=str, help='name of predictions file')
parser.add_argument("--dataset", default='RAF', type=str, help='selects dataset (RAF)')
args = parser.parse_args()

"""
General setup
"""
# Set data dir
data_dir = "./data"
raf_dir = os.path.join(data_dir, args.dataset)

# Set predictions file path
predictions_file = os.path.join(args.predictions_dir, args.predictions_file)

# Get filename lookup table
lookup_table = RAF.get_lookup_table(os.path.join(raf_dir, "splits/all_files.csv"))

"""
Load predictions
"""
# See predict.py for npz save format
predictions = np.load(predictions_file)
ys = predictions["test_ys"]
yhats = predictions["test_yhats"]
ys_c = np.argmax(ys, axis=1)
yhats_c = np.argmax(yhats, axis=1)
filenames = predictions["test_filenames"]
n = np.shape(ys)[0]

# Store protected attributes
races = np.array([lookup_table[fn]["Race"] for fn in filenames])
genders = np.array([lookup_table[fn]["Gender"] for fn in filenames])
ages = np.array([lookup_table[fn]["Age"] for fn in filenames])

"""
Evaluation metrics
"""
def eval(dict, yhats_col, yhats_arr, ys_col, ys_arr):
    """Define all the metrics"""
    def acc(yhats_col, ys_col):
        return np.sum(yhats_col == ys_col) / np.shape(yhats_col)[0]
    def f1(yhats_col, ys_col, average="weighted"):
        return f1_score(ys_col, yhats_col, labels=list(range(7)), average=average)
    def auc(yhats_arr, ys_arr, multi_class="ovr"):
        return roc_auc_score(ys_arr, yhats_arr, multi_class=multi_class)
    def unfairness(yhats_col, ys_col, attr):
        """
        Denis et al, 2022, arxiv
        I generalized their defn to apply to |S|>2
        """
        U = []
        for k in range(7):
            probs = []
            mask = None
            if attr == "race":
                for r in range(3):
                    mask = (races == r)
                    probs.append(
                        np.sum(np.logical_and(yhats_col[mask] == k, ys_col[mask] == k)) / np.sum(ys_col[mask] == k))
            elif attr == "gender":
                for g in range(2):  # only use male/female since there aren't unknown samples for every emotion
                    mask = (genders == g)
                    probs.append(
                        np.sum(np.logical_and(yhats_col[mask] == k, ys_col[mask] == k)) / np.sum(ys_col[mask] == k))
            elif attr == "age":
                for a in range(1, 5): # only use middle three age groups since there aren't unknown samples for every emotion
                    mask = (ages == a)
                    probs.append(
                        np.sum(np.logical_and(yhats_col[mask] == k, ys_col[mask] == k)) / np.sum(ys_col[mask] == k))
            else: raise NotImplementedError()
            probs = np.array(probs)
            U.append(
                np.max(probs) - np.min(probs)
            )
        return np.max(np.array(U))

    """Iterate over attribute combinations"""
    # overall
    dict["acc"] = acc(yhats_col, ys_col)
    dict["f1"] = f1(yhats_col, ys_col)
    dict["auc"] = auc(yhats_arr, ys_arr)
    dict["unfairness_race"] = unfairness(yhats_col, ys_col, attr="race")
    dict["unfairness_gender"] = unfairness(yhats_col, ys_col, attr="gender")
    dict["unfairness_age"] = unfairness(yhats_col, ys_col, attr="age")
    # within each race
    for r in range(3):
        mask = (races == r)
        dict[f"acc_race{r}"] = acc(yhats_col[mask], ys_col[mask])
    # within each gender
    for g in range(3):
        mask = (genders == g)
        dict[f"acc_gender{g}"] = acc(yhats_col[mask], ys_col[mask])
    # within each age
    for a in range(5):
        mask = (ages == a)
        dict[f"acc_age{a}"] = acc(yhats_col[mask], ys_col[mask])
    # within each race & gender combination
    for r in range(3):
        for g in range(3):
            mask = ((races == r) & (genders == g))
            dict[f"acc_race{r}_gender{g}"] = acc(yhats_col[mask], ys_col[mask])
    # within each race & age combination
    for r in range(3):
        for a in range(5):
            mask = ((races == r) & (ages == a))
            dict[f"acc_race{r}_age{a}"] = acc(yhats_col[mask], ys_col[mask])
    # within each gender & age combination
    for g in range(3):
        for a in range(5):
            mask = ((genders == g) & (ages == a))
            dict[f"acc_gender{g}_age{a}"] = acc(yhats_col[mask], ys_col[mask])
    # within each gender & age & race combination
    for r in range(3):
        for g in range(3):
            for a in range(5):
                mask = ((races == r) & (genders == g) & (ages == a))
                if np.sum(mask) == 0:
                    dict[f"acc_race{r}_gender{g}_age{a}"] = np.nan
                    continue
                dict[f"acc_race{r}_gender{g}_age{a}"] = acc(yhats_col[mask], ys_col[mask])

    """Return dictionary"""
    return dict

stats = {}
stats = eval(stats, yhats_c, yhats, ys_c, ys)

"""
Print to console.
"""
for k in stats.keys():
    print(f"{k}: {stats[k]}")