from __future__ import print_function, absolute_import

import os
import time
import pickle
import pathlib
import copy
import argparse
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter

from data_clothing1M import clothing_dataloader
from losses import WeightedCCE

# set seed for reproducibility
torch.manual_seed(123)
torch.cuda.manual_seed_all(123)
np.random.seed(123)

torch.autograd.set_detect_anomaly(True)

EPS = 1e-8


PARSER = argparse.ArgumentParser(description='PyTorch Clothing-1M Batch_Rewgt Training')
PARSER.add_argument('-loss', '--loss_name', default="cce", type=str, help="loss name")
PARSER.add_argument("-bs", "--batch_size", default=128, type=int, help="batch size")
PARSER.add_argument('-ep', '--num_epoch', default=15, type=int, help="number of epochs")
PARSER.add_argument('-run', '--num_runs', default=1, type=int, help="number of runs/simulations")
ARGS = PARSER.parse_args()


def accuracy(true_label, pred_label):
    num_samples = true_label.shape[0]
    err = [1 if (pred_label[i] != true_label[i]).sum() == 0 else 0 for i in range(num_samples)]
    acc = 1 - (sum(err)/num_samples)
    return acc


def batch_rewgt_train(dat_loader, net):

    loss_train_loc = 0.
    acc_train_loc = 0.
    correct = 0

    net.train()

    for batch_id, (x, y) in tqdm(enumerate(dat_loader)):
        y = y.type(torch.LongTensor)
        # Transfer data to the GPU
        x, y = x.to(DEVICE), y.to(DEVICE)

        # x = x.reshape((-1, 784))

        output = net(x)
        pred_prob = F.softmax(output, dim=1)
        pred = torch.argmax(pred_prob, dim=1)

        # batch_loss = nn.CrossEntropyLoss(reduction='mean')
        # loss_batch = batch_loss(output, y)
        loss_batch, _ = loss_fn(output, y)

        optimizer.zero_grad()
        loss_batch.mean().backward()
        optimizer.step()

        # .item() for scalars, .tolist() in general
        # loss_train_loc += (torch.mean(batch_loss(output, y.to(DEVICE)))).item()

        # loss_train_loc += torch.mean(loss_fn(output, y.to(DEVICE))).item()
        loss_train_loc += torch.mean(loss_batch).item()
        correct += (pred.eq(y.to(DEVICE))).sum().item()

        batch_cnt = batch_id + 1

    loss_train_loc /= batch_cnt
    acc_train_loc = 100.*correct/len(dat_loader.dataset)

    return loss_train_loc, acc_train_loc

def test(dat_loader, net, run_num, use_best=False):

    loss_test_loc = 0.
    acc_test_loc = 0.
    correct = 0

    net.eval()

    with torch.no_grad():
        for batch_id, (x, y) in enumerate(dat_loader):
            if use_best == True:
                # load best model weights
                net.load_state_dict(torch.load(chkpt_path + f"{MODE}-{DATASET}-{LOSS_NAME}-mdl-\
                                                              wts-run-{str(run_num)}.pt"))
                net = net.to(DEVICE)

            y = y.type(torch.LongTensor)
            x, y = x.to(DEVICE), y.to(DEVICE)
            output = net(x)
            pred_prob = F.softmax(output, dim=1)
            pred = torch.argmax(pred_prob, dim=1)

            loss_batch, _ = loss_fn(output, y)
            loss_test_loc += torch.mean(loss_batch).item()
            correct += (pred.eq(y.to(DEVICE))).sum().item()

            batch_cnt = batch_id + 1
    loss_test_loc /= batch_cnt
    acc_test_loc = 100.*correct/len(dat_loader.dataset)

    return loss_test_loc, acc_test_loc


T_START = time.time()

"""
Configuration
"""

# random_state = 422
DATASET = "clothing1M"
BATCH_SIZE = ARGS.batch_size
LOSS_NAME = ARGS.loss_name
NUM_EPOCH = ARGS.num_epoch
NUM_RUNS = ARGS.num_runs
LEARNING_RATE = 1e-3
MODE = "batch_rewgt_clothing1M"
NUM_CLASS = 14

"""
Loss Function
"""

if LOSS_NAME == "cce":
    loss_fn = WeightedCCE(k=1, num_class=NUM_CLASS, reduction="none")
else:
    raise NotImplementedError(f"Batch Reweighting not implemented for - {LOSS_NAME}")

print("\n==============\nLOSS_NAME: {}\n=============\n".format(LOSS_NAME))


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


for run in range(NUM_RUNS):

    T_START = time.time()

    epoch_loss_train = []
    epoch_acc_train = []
    epoch_loss_test = []
    epoch_acc_test = []

    best_acc_val = 0.

    chkpt_path = f"./checkpoint/{MODE}/{DATASET}/run_{str(run)}/"

    res_path = f"./results_pkl/{MODE}/{DATASET}/run_{str(run)}/"

    plt_path = f"./plots/{MODE}/{DATASET}/run_{str(run)}/"

    log_dirs_path = f"./runs/{MODE}/{DATASET}/run_{str(run)}/"

    if not os.path.exists(chkpt_path):
        os.makedirs(chkpt_path)

    if not os.path.exists(res_path):
        os.makedirs(res_path)

    if not os.path.exists(plt_path):
        os.makedirs(plt_path)

    if not os.path.exists(log_dirs_path):
        os.makedirs(log_dirs_path)
    else:
        for f in pathlib.Path(log_dirs_path).glob('events.out*'):
            try:
                f.unlink()
            except OSError as e:
                print(f"Error: {f} : {e.strerror}")
        print("\nLog files cleared...\n")

    print("\n============ PATHS =================\n")
    print(f"chkpt_path: {chkpt_path}")
    print(f"res_path: {res_path}")
    print(f"plt_path: {plt_path}")
    print(f"log_dirs_path: {log_dirs_path}")
    print("file name: " + f"{MODE}-{DATASET}-{LOSS_NAME}-run-{str(run)}.pt")
    print("\n=============================\n")


    """
    Training/Validation/Test Data
    """

    # # dat, ids = read_data(noise_type, noise_rate, DATASET, data_aug, MODE)

    # # X_temp, y_temp, X_train, y_train = dat[0], dat[1], dat[2], dat[3]
    # # X_val, y_val, X_test, y_test = dat[4], dat[5], dat[6], dat[7]
    # # idx, idx_train, idx_val = ids[0], ids[1], ids[2]


    # print("\n=============================\n")
    # print("X_train: ", X_train.shape, " y_train: ", y_train.shape, "\n")
    # print("X_val: ", X_val.shape, " y_val: ", y_val.shape, "\n")
    # print("X_test: ", X_test.shape, " y_test: ", y_test.shape, "\n")
    # print("\n=============================\n")

    """
    Create Dataset Loader
    """

    data_loader = clothing_dataloader(batch_size=BATCH_SIZE, shuffle=True)
    train_loader, val_loader, test_loader = data_loader.run()

    print(train_loader.dataset, "\n", len(train_loader.dataset), "\n")

    input("\nPress <ENTER> to continue.\n")

    """
    Initialize n/w and optimizer
    """

    model = models.resnet50(pretrained=True)
    model.fc = nn.Linear(2048, NUM_CLASS)
    model = model.to(DEVICE)
    print(model)

    """
    Optimizer
    """
    # optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=1e-3)
    # lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[6, 11], gamma=0.5)


    """
    Use this optimizer and data config. for Clothing-1M, Animal-10N, Food-101N
    https://openreview.net/pdf?id=ZPa2SyGcbwh
    """

    """
    Setting up Tensorbard
    """
    writer = SummaryWriter(log_dirs_path)

    for epoch in range(NUM_EPOCH):

        #Training set performance
        loss_train, acc_train = batch_rewgt_train(train_loader, model)
        writer.add_scalar('training_loss', loss_train, epoch)
        writer.add_scalar('training_accuracy', acc_train, epoch)
        writer.close()
        # Validation set performance
        loss_val, acc_val = test(val_loader, model, run, use_best=False)
        #Testing set performance
        loss_test, acc_test = test(test_loader, model, run, use_best=False)
        writer.add_scalar('testing_loss', loss_test, epoch)
        writer.add_scalar('testing_accuracy', acc_test, epoch)
        writer.close()

        # if epoch >= 40:
        #     LEARNING_RATE /= 10

        # for param_group in optimizer.param_groups:
        #     param_group['lr'] = LEARNING_RATE

        lr_scheduler.step()

        epoch_loss_train.append(loss_train)
        epoch_acc_train.append(acc_train)
        epoch_loss_test.append(loss_test)
        epoch_acc_test.append(acc_test)

        # Update best_acc_val
        if epoch == 0:
            best_acc_val = acc_val


        if acc_val > best_acc_val:
            best_acc_val = acc_val
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), chkpt_path + f"{MODE}-{DATASET}-{LOSS_NAME}-mdl\
                       -wts-run-{str(run)}.pt")
            print("Model weights updated...\n")

        print("Epoch: {}, lr: {}, loss_train: {}, loss_val: {}, loss_test: {:.3f}, acc_train: {},\
               acc_val: {}, acc_test: {:.3f}\n".format(epoch, optimizer.param_groups[0]['lr'],
                                                       loss_train, loss_val, loss_test,
                                                       acc_train, acc_val, acc_test))


    # Test accuracy on the best_val MODEL
    loss_test, acc_test = test(test_loader, model, run, use_best=False)
    print("Test set performance - test_acc: {}, test_loss: {}\n".format(acc_test, loss_test))

    # Print the elapsed time
    ELAPSED = time.time() - T_START
    print("\nelapsed time: \n", ELAPSED)

    """
    Save results
    """
    with open(res_path+ f"{MODE}-{DATASET}-{LOSS_NAME}-run-{str(run)}.pickle", 'wb') as f:
        pickle.dump({'epoch_loss_train': np.asarray(epoch_loss_train),
                     'epoch_acc_train': np.asarray(epoch_acc_train),
                     'epoch_loss_test': np.asarray(epoch_loss_test),
                     'epoch_acc_test': np.asarray(epoch_acc_test),
                     'num_epoch': NUM_EPOCH,
                     'time_elapsed': ELAPSED},
                    f, protocol=pickle.HIGHEST_PROTOCOL)
        print("Pickle file saved: " + res_path+ f"{MODE}-{DATASET}-{LOSS_NAME}-\
               run-{str(run)}.pickle", "\n")
