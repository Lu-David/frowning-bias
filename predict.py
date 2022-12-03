"""
Get model predictions for a test set.
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
from sklearn.metrics import roc_auc_score
import custom_modules as RAF

# Parse args
parser = argparse.ArgumentParser()
parser.add_argument("--model-dir", default='./results/scratch', type=str, help='directory where model is stored')
parser.add_argument("--model-file", type=str, help='name of pt file for model')
parser.add_argument("--model-name", type=str, help='model name for saving')
parser.add_argument("--architecture", default='ResNet18', type=str, help='selects architecture (ResNet18/***)')
parser.add_argument("--pretrained", default='y', type=str, help='use pretrained model (y/n)')
parser.add_argument("--frozen", default='n', type=str, help='freeze all but task head (y/n)')
parser.add_argument("--batch-size", default=16, type=int, help='batch size')
parser.add_argument("--dataset", default='RAF', type=str, help='selects dataset (RAF)')
parser.add_argument("--test-file", default='test_files.csv', type=str, help='name of csv file to use as test list')
parser.add_argument("--use-parallel", default='y', type=str, help='***')
parser.add_argument("--num-workers", default=12, type=int, help='number of dataloader workers')
parser.add_argument("--image-size", default=224, type=int, help='image size to use')
parser.add_argument("--print-batches", default='n', type=str, help='print batch updates')
parser.add_argument("--scratch-dir", default='~/Documents/scratch', type=str, help='scratch dir for tmp files')
parser.add_argument("--results-dir", default='./RAF_results', type=str, help='directory to save results')
parser.add_argument("--use-gpus", default='all', type=str, help='gpu ids (comma separated list eg "0,1" or "all")')
args = parser.parse_args()

"""
General setup
"""
# Set GPU vis
if args.use_gpus != 'all':
    os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES']=args.use_gpus

# Set data dir
data_dir = "./data"

# Set model path
model_path = os.path.join(args.model_dir, args.model_file)

# Parse args
args.pretrained = args.pretrained == 'y'
n_labels = 7 # see utils
args.frozen = args.frozen == 'y'
args.use_parallel = args.use_parallel == 'y'
args.print_batches = args.print_batches == 'y'
args.scratch_dir = args.scratch_dir.replace('~',os.path.expanduser('~'))
args.results_dir = args.results_dir.replace('~',os.path.expanduser('~'))

# Model params
model_args = {
    'model_type': args.architecture,
    'pretrained': args.pretrained,
    'n_labels': n_labels,
    'frozen': args.frozen,
    'batch_size': args.batch_size,
    'data_dir': data_dir,
    'dataset': args.dataset,
    'test_file': args.test_file,
    'use_parallel': args.use_parallel,
    'num_workers': args.num_workers,
    'img_size': args.image_size,
    'print_batches': args.print_batches,
    'scratch_dir':args.scratch_dir,
    'results_dir':args.results_dir,
    'results_file': os.path.join(args.results_dir, '{}_predict{}.npz'.format(args.model_name, args.test_file))
}

# Save this to a prediction log
prediction_log_path = "./results/predictions_log.txt"
with open(prediction_log_path, "a") as f:
    f.write(f"{args.model_name} , {args.model_dir} , {args.model_file} , {model_args['results_file']}\n")

print(model_args)

"""
Model setup
"""
# Setup
model = RAF.load_model(model_path, model_args)
print("Model loaded.")

# Datasets
dataset_root = os.path.join(model_args['data_dir'], model_args['dataset'])
test_data = RAF.RAFDataset(
    csv_file=os.path.join(dataset_root, "splits/{}".format(args.test_file)),
    n_labels=model_args['n_labels'],
    img_size=model_args["img_size"],
    transform=None,
)

# Get device
device = 'cpu'
ncpus = os.cpu_count()
dev_n = ncpus
if torch.cuda.is_available():
    device = 'cuda'
    dev_n = torch.cuda.device_count()
print('Device: {} #: {} #cpus: {}\n'.format(device, dev_n, ncpus))

# Data loaders
testLoader = DataLoader(test_data, batch_size=model_args['batch_size'],
                         pin_memory=True, shuffle=True,
                         num_workers=model_args['num_workers'])

# Loss function
# multi-class, expects unnormalized logits
loss_fxn = nn.CrossEntropyLoss()

  
"""
Predictions
"""

if model_args['use_parallel']:
  model = nn.DataParallel(model)

# Model to device
model = model.to(device)

# Time it
time_start = time.time()

# Val
model.eval()

test_timer = 0
test_loss = 0
test_auc = 0
batch_counter = 1
test_ys = []
test_yhats = []
test_filenames = []

# For each batch
for x, y, filename, _ in testLoader:
    if model_args['print_batches']:
        print('Batch {}/{}'.format(batch_counter, len(testLoader)))
    batch_counter += 1

    with torch.no_grad():
        x = x.to(device)
        y = y.to(device)

        yhat = model(x)
        loss = loss_fxn(yhat, y)
        
        # normalize yhats before saving
        yhat = T.nn.functional.softmax(yhat)

        test_loss += loss.item() / len(testLoader)
        test_ys.extend(y.to('cpu').numpy().tolist())
        test_yhats.extend(yhat.to('cpu').numpy().tolist())
        test_filenames.extend(filename)

# Stats
test_timer = (time.time() - time_start) / 60

# Multiclass AUC (one-v-rest), expects normalized predictions
test_auc = roc_auc_score(test_ys, test_yhats, multi_class='ovr')

# Accuracy
test_yhats_arr = np.array(test_yhats)
test_ys_arr = np.array(test_ys)
test_acc = np.sum(np.argmax(test_yhats_arr, axis=1) == np.argmax(test_ys_arr, axis=1)) / len(test_data)

# Print
print('Test loss: {:.4f} Test AUC: {:.4f} Test Acc: {:.4f} Time (min): {:.2f}'.format(
        test_loss,
        test_auc,
        test_acc,
        test_timer))

"""
Save predictions
"""
np.savez(
    model_args['results_file'],
    test_yhats=test_yhats_arr, test_ys=test_ys_arr, test_filenames=test_filenames
)
