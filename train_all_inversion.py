"""
Trains CNN model on RAF. Command line parameters.
Uses original (i.e. all of the available) training data, test data for stopping.
"""


# Imports
import os
from pathlib import Path

import numpy as np
import time
import argparse
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
import custom_modules as RAF


def main():
    # Parse args
    parser = argparse.ArgumentParser()
    parser.add_argument("--architecture", default='ResNet18', type=str, help='selects architecture (ResNet18/***)')
    parser.add_argument("--pretrained", default='y', type=str, help='use pretrained model (y/n)')
    parser.add_argument("--frozen", default='n', type=str, help='freeze all but task head (y/n)')
    parser.add_argument("--initial-lr", default=1e-2, type=float, help='initial learning rate')
    parser.add_argument("--batch-size", default=16, type=int, help='batch size')
    parser.add_argument("--max-epochs", default=50, type=int, help='stop training after this many epochs')
    parser.add_argument("--optimizer-family", default='AdamW', type=str, help='optimizer to use (SGD/AdamW)')
    parser.add_argument("--weight-decay", default=1e-4, type=float, help='weight decay')
    parser.add_argument("--momentum", default=0.9, type=float, help='momentum')
    parser.add_argument("--scheduler-family", default='step', type=str, help='selects scheduler family (step/drop)')
    parser.add_argument("--drop-factor", default=0.1, type=float, help='drop factor for scheduler')
    parser.add_argument("--plateau-patience", default=3, type=int, help='number of epochs to wait for improvement for scehduler')
    parser.add_argument("--plateau-threshold", default=1e-4, type=float, help='minimum change to count as improvement')
    parser.add_argument("--break-patience", default=5, type=int, help='number of epochs to wait for improvement before stopping training')
    parser.add_argument("--dataset", default='RAF', type=str, help='selects dataset (RAF)')
    parser.add_argument("--train-file", default='train_files.csv', type=str, help='name of csv file to use as training list')
    parser.add_argument("--val-file", default='val_files.csv', type=str, help='name of csv file to use as val list')
    parser.add_argument("--use-parallel", default='y', type=str, help='***')
    parser.add_argument("--train-transform", default='none', type=str, help='selects transforms to use in training (none/***)')
    parser.add_argument("--num-workers", default=12, type=int, help='number of dataloader workers')
    parser.add_argument("--dropout", default=0, type=float, help='dropout')
    parser.add_argument("--image-size", default=224, type=int, help='image size to use')
    parser.add_argument("--class-weights", default='y', type=str, help='class weights for loss function at train time (y/n)')
    parser.add_argument("--equalized-by", default='none', type=str, help='equalize training set by a protected attr (Race/Gender/Age/none)')
    parser.add_argument("--equalized-how", default='none', type=str, help='equalize training set by sampling attr (up/down/none)')
    parser.add_argument("--print-batches", default='n', type=str, help='print batch updates')
    parser.add_argument("--scratch-dir", default='~/Documents/scratch', type=str, help='scratch dir for tmp files')
    parser.add_argument("--results-dir", default='./results/scratch', type=str, help='directory to save results')
    parser.add_argument("--results-file", default='', type=str, help='directory to save results')
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

    # Parse args
    args.pretrained = args.pretrained == 'y'
    n_labels = 7 # see utils
    args.frozen = args.frozen == 'y'
    args.class_weights = args.class_weights == 'y'
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
        'initial_lr': args.initial_lr,
        'batch_size': args.batch_size,
        'max_epochs': args.max_epochs,
        'optimizer_family': args.optimizer_family,
        'weight_decay': args.weight_decay,
        'momentum': args.momentum,
        'scheduler_family': args.scheduler_family,
        'drop_factor': args.drop_factor,
        'plateau_patience': args.plateau_patience,
        'plateau_threshold': args.plateau_threshold,
        'break_patience': args.break_patience,
        'data_dir': data_dir,
        'dataset': args.dataset,
        'train_file': args.train_file,
        'val_file': args.val_file,
        'use_parallel': args.use_parallel,
        'train_transform': args.train_transform,
        'num_workers': args.num_workers,
        'dropout': args.dropout,
        'img_size': args.image_size,
        'class_weights': args.class_weights,
        'equalized_by': args.equalized_by,
        'equalized_how': args.equalized_how,
        'print_batches': args.print_batches,
        'scratch_dir':args.scratch_dir,
        'results_dir':args.results_dir,
        'results_file': '{}_lr{}_bs{}_opt{}_wd{}_sch_{}_pp{}_bp{}_tr{}_va{}_tf{}_do{}_cw{}_eq{}_eqhow{}_{}.txt'.format(
            args.architecture, args.initial_lr, args.batch_size, args.optimizer_family,
            args.weight_decay, args.scheduler_family, args.plateau_patience, args.break_patience,
            args.train_file, args.val_file, args.train_transform, args.dropout, args.class_weights, args.equalized_by,
            args.equalized_how, int(time.time()))[:-4] if args.results_file == '' else args.results_file
    }

    # Print fxn
    def printToResults(to_print,file_name):
        with open(file_name, 'a') as f:
            f.write(to_print)

    print(model_args)

    Path(model_args['results_dir']).mkdir(parents=True, exist_ok=True)

    """
    Model setup
    """
    # Setup
    model = RAF.get_model(model_args)

    # Set dropout
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.p = model_args['dropout']

    # Datasets (original train/test)
    dataset_root = os.path.join(model_args['data_dir'], model_args['dataset'])
    train_data = RAF.RAFDatasetAugmented(
        csv_file=os.path.join(dataset_root, "splits/original_train_files.csv"),
        n_labels=model_args['n_labels'],
        img_size=model_args["img_size"],
        transform=None,
        equalized_by=model_args['equalized_by'],
        equalized_how=model_args['equalized_how']
    )
    val_data = RAF.RAFDatasetAugmented(
        csv_file=os.path.join(dataset_root, "splits/test_files.csv"),
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
    trainLoader = DataLoader(train_data, batch_size=model_args['batch_size'],
                             pin_memory=True, shuffle=True,
                             num_workers=model_args['num_workers'])

    valLoader = DataLoader(val_data, batch_size=model_args['batch_size'],
                           pin_memory=True, shuffle=True,
                           num_workers=model_args['num_workers'])

    # Loss function
    # multi-class, expects unnormalized logits
    loss_fxn = nn.CrossEntropyLoss()
    if model_args['class_weights']:
        # weight each class
        # using inverse of # of samples in each class based on train dataset
        train_label_dist = train_data.label_distribution()
        weight = np.array([1/train_label_dist[x] for x in range(7)])
        weight = torch.from_numpy(weight).to(device)
        # normalize
        weight = torch.nn.functional.softmax(weight)
        loss_fxn = nn.CrossEntropyLoss(weight=weight)

    # Optimizer
    optimizer = None
    if model_args['optimizer_family'] == 'SGD':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=model_args['initial_lr'],
            momentum=model_args['momentum'],
            weight_decay=model_args['weight_decay']
        )
    elif model_args['optimizer_family'] == 'AdamW':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=model_args['initial_lr'],
            weight_decay=model_args['weight_decay']
        )

    # Scheduler
    scheduler = None
    if model_args['scheduler_family'] == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=optimizer,
            step_size=model_args['plateau_patience'],
            gamma=model_args['drop_factor'],
            verbose=False
        )
    elif model_args['scheduler_family'] == 'drop':
      scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        factor=model_args['drop_factor'],
        patience=model_args['plateau_patience'],
        verbose=False
      )


    """
    Train loop
    """

    # Train logs
    best_log = {'epoch': -1,
                'loss': {'train': 999999, 'val': 999999},
                'auc': {'train': 0, 'val': 0},
                'acc': {'train': 0.0, 'val': 0.0},
                'points': {'train': {'y': [], 'yhat': []}, 'val': {'y': [], 'yhat': []}},
                'timer': 0
                }
    train_log = {'epoch': -1,
                 'loss': {'train': [], 'val': []},
                 'auc': {'train': [], 'val': []},
                 'acc': {'train': [], 'val': []},
                 'timer': []
                 }

    if model_args['use_parallel']:
      model = nn.DataParallel(model)

    # Model to device
    model = model.to(device)

    # Epoch loop
    for epoch in range(model_args['max_epochs']):
        time_start = time.time()
        train_log['epoch'] = epoch

        # Train
        model.train()

        train_loss = 0
        batch_counter = 0
        train_ys = []
        train_yhats = []

        for x, y, fns, attrs in trainLoader:
            if model_args['print_batches']:
                print('\rEpoch {}\t{} batch {}/{}'.format(epoch, 'train', batch_counter, len(trainLoader)))
            batch_counter += 1

            x = x.to(device)
            y = y.to(device)

            yhat = model(x)
            loss = loss_fxn(yhat, y)


            # normalize yhats before saving
            yhat = torch.nn.functional.softmax(yhat)

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            with torch.no_grad():
                train_loss += loss.item() / len(trainLoader)
                train_ys.extend(y.to('cpu').numpy().tolist())
                train_yhats.extend(yhat.to('cpu').numpy().tolist())

        # Val
        model.eval()

        val_loss = 0
        batch_counter = 0
        val_ys = []
        val_yhats = []

        # For each batch
        for x, y, fns, attrs in valLoader:
            if model_args['print_batches']:
                print('Epoch {}\t{} batch {}/{}'.format(epoch, 'val', batch_counter, len(valLoader)))
            batch_counter += 1

            with torch.no_grad():
                x = x.to(device)
                y = y.to(device)

                yhat = model(x)
                loss = loss_fxn(yhat, y)


                # normalize yhats before saving
                yhat = torch.nn.functional.softmax(yhat)

                val_loss += loss.item() / len(valLoader)
                val_ys.extend(y.to('cpu').numpy().tolist())
                val_yhats.extend(yhat.to('cpu').numpy().tolist())

        # Add to train_log
        epoch_time = (time.time() - time_start) / 60
        train_log['timer'].append(epoch_time)
        train_log['loss']['train'].append(train_loss)
        train_log['loss']['val'].append(val_loss)
        train_log['acc']['train'].append(
            np.sum(np.argmax(train_yhats, axis=1) == np.argmax(train_ys, axis=1)) / np.shape(train_ys)[0]
        )
        train_log['acc']['val'].append(
            np.sum(np.argmax(val_yhats, axis=1) == np.argmax(val_ys, axis=1)) / np.shape(val_ys)[0]
        )

        # Multiclass AUC (one-v-rest), expects normalized predictions
        train_log['auc']['train'].append(roc_auc_score(train_ys, train_yhats, multi_class='ovr'))
        train_log['auc']['val'].append(roc_auc_score(val_ys, val_yhats, multi_class='ovr'))

        # Best
        # Update best based on auc
        if train_log['auc']['val'][-1] - best_log['auc']['val'] >= model_args['plateau_threshold']:
            # Print
            print('New best!')

            # Update best arg
            best_log = {'epoch': epoch,
                        'loss': {'train': train_loss, 'val': val_loss},
                        'auc': {'train': train_log['auc']['train'][-1], 'val': train_log['auc']['val'][-1]},
                        'acc': {'train': train_log['acc']['train'][-1], 'val': train_log['acc']['val'][-1]},
                        'points': {'train': {'y': train_ys, 'yhat': train_yhats},
                                   'val': {'y': val_ys, 'yhat': val_yhats}},
                        'timer': sum(train_log['timer'])
                        }

            # Save
            torch.save(model.module.state_dict(), os.path.join(model_args['results_dir'], model_args['results_file']+'_model.pt'))
            torch.save(best_log, os.path.join(model_args['results_dir'], model_args['results_file']+'_stats.pt'))

        # Print
        print('Epoch {}\tTrain loss: {:.4f} Val loss: {:.4f} Train AUC: {:.4f} Val AUC: {:.4f} Train acc: {:.4f} Val acc: {:.4f} Time (min): {:.2f} Total time: {:.2f}'.format(
                epoch,
                train_log['loss']['train'][-1],
                train_log['loss']['val'][-1],
                train_log['auc']['train'][-1],
                train_log['auc']['val'][-1],
                train_log['acc']['train'][-1],
                train_log['acc']['val'][-1],
                epoch_time,
                sum(train_log['timer'])))

        if epoch - best_log['epoch'] > model_args['break_patience']:
            print('Breaking epoch loop')
            break

        # LR Scheduler step
        if model_args['scheduler_family'] == 'no-scheduler':
            pass
        elif model_args['scheduler_family'] == 'drop':
            scheduler.step(train_log['loss']['val'][-1])
        else:
            scheduler.step()

    """
    Post train
    """
    epoch = best_log['epoch']
    train_loss = best_log['loss']['train']
    val_loss = best_log['loss']['val']
    val_auc = best_log['auc']['val']
    train_acc = best_log['acc']['train']
    val_acc = best_log['acc']['val']

    results = {
        'File': model_args['results_file'],
        'Architecture': model_args['model_type'],
        '% Data': model_args['train_file'][model_args['train_file'].rfind('_') + 1:model_args['train_file'].rfind('.')],
        'Initial LR': model_args['initial_lr'],
        'Optimizer': model_args['optimizer_family'],
        'Scheduler': model_args['scheduler_family'],
        'Scheduler Patience': model_args['plateau_patience'],
        'Break Patience': model_args['break_patience'],
        'Scheduler Drop Factor': model_args['drop_factor'],
        'Batch Size': model_args['batch_size'],
        'Weight Decay': model_args['weight_decay'],
        'Frozen': model_args['frozen'],
        'Transform': model_args['train_transform'],
        'Dropout': model_args['dropout'],
        'Class Weights': model_args['class_weights'],
        'Equalized By': model_args['equalized_by'],
        'Equalized How': model_args['equalized_how'],
        'Epoch': epoch,
        'Loss Train': train_loss,
        'Loss Val': val_loss,
        'AUC Val': val_auc,
        'Accuracy Train': train_acc,
        'Accuracy Val': val_acc,
        'Total Time': best_log['timer']
    }

    print(','.join([str(x) for x in results.values()]))
    printToResults(','.join([str(x) for x in results.values()])+'\n', os.path.join(model_args['results_dir'],'results.csv'))

if __name__ == '__main__':
    main()