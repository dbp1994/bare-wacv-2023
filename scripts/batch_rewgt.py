from __future__ import print_function, absolute_import

import os
import time
import argparse
import pickle
import pathlib
import copy

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter

from data import read_data
from losses import WeightedCCE

# set seed for reproducibility
torch.manual_seed(1337)
np.random.seed(3459)
# tf.set_random_seed(3459)


torch.autograd.set_detect_anomaly(True)

EPS = 1e-8


PARSER = argparse.ArgumentParser(description='PyTorch MNIST Batch_Rewgt Training')
PARSER.add_argument('-dat', '--dataset', default="mnist", type=str, help="dataset")
PARSER.add_argument('-nr', '--noise_rate', default=0.4, type=float, help="noise rate")
PARSER.add_argument('-nt', '--noise_type', default="sym", type=str, help="noise type")
PARSER.add_argument('-loss', '--loss_name', default="cce", type=str, help="loss name")
PARSER.add_argument('-da', '--data_aug', default=0, type=int, help="data augmentation (0 or 1)")
PARSER.add_argument("-bs", "--batch_size", default=128, type=int, help="batch size")
PARSER.add_argument('-ep', '--num_epoch', default=100, type=int, help="number of epochs")
PARSER.add_argument('-run', '--num_runs', default=1, type=int, help="number of runs/simulations")
ARGS = PARSER.parse_args()


def accuracy(true_label, pred_label):
    """
    calculate accuracy
    """
    num_samples = true_label.shape[0]
    err_frac = [1 if (pred_label[i] != true_label[i]).sum() == 0 else 0 for i in range(num_samples)]
    acc = 1 - (sum(err_frac)/num_samples)
    return acc

def batch_rewgt_train(dat_loader, net):

    loss_train_loc = 0.
    acc_train_loc = 0.
    correct = 0

    idx_sel_tr_agg_loc = np.zeros((len(dat_loader.dataset), ))

    net.train()

    for batch_id, (x, y, idx) in tqdm(enumerate(dat_loader)):

        y = y.type(torch.LongTensor)
        # Transfer data to the GPU
        x, y, idx = x.to(DEVICE), y.to(DEVICE), idx.to(DEVICE)
        output = net(x)
        pred_prob = F.softmax(output, dim=1)
        pred = torch.argmax(pred_prob, dim=1)

        loss_batch, idx_sel = loss_fn(output, y)

        optimizer.zero_grad()
        loss_batch.mean().backward()
        optimizer.step()

        loss_train_loc += torch.mean(loss_batch).item()
        correct += (pred.eq(y.to(DEVICE))).sum().item()

        idx_tmp = torch.zeros(x.shape[0])
        idx_tmp[idx_sel] = 1.
        idx_sel_tr_agg_loc[list(map(int, idx.tolist()))] = np.asarray(idx_tmp.tolist())

        batch_cnt = batch_id + 1

    loss_train_loc /= batch_cnt
    acc_train_loc = 100.*correct/len(dat_loader.dataset)

    return loss_train_loc, acc_train_loc, idx_sel_tr_agg_loc

def test(data_loader, net, run, use_best=False):

    loss_test = 0.
    correct = 0

    net.eval()

    with torch.no_grad():
        for batch_id, (x, y) in enumerate(data_loader):
            if use_best == True:
                # load best model weights
                net.load_state_dict(torch.load(chkpt_path + f"{MODE}-{DATASET}-{loss_name}-\
                    {NOISE_TYPE}-nr-0{str(int(NOISE_RATE * 10))}-mdl-wts-run-{str(run)}.pt"))
                net = net.to(DEVICE)

            y = y.type(torch.LongTensor)
            x, y = x.to(DEVICE), y.to(DEVICE)
            output = net(x)
            pred_prob = F.softmax(output, dim=1)
            pred = torch.argmax(pred_prob, dim=1)

            loss_batch, _ = loss_fn(output, y)
            loss_test += torch.mean(loss_batch).item()
            correct += (pred.eq(y.to(DEVICE))).sum().item()
            batch_cnt = batch_id + 1
    loss_test /= batch_cnt
    acc_test = 100.*correct/len(data_loader.dataset)
    return loss_test, acc_test


# Model
class MNIST_NN(nn.Module):
    def __init__(self, temp=1.0):
        super(MNIST_NN, self).__init__()

        # 1 I/P channel, 6 O/P channels, 5x5 conv. kernel
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5)

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # self.fc1 = nn.Linear(400, 120)
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, 10)
        self.temp = temp
    def forward(self, x):

        # x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2), stride=2)
        # x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2), stride=2)
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x)) / self.temp
        return x

    def num_flat_features(self, x):
        size_tot = x.size()[1:] # all dims except batch_size dim
        num_features = 1
        for size_dim in size_tot:
            num_features *= size_dim
        return num_features


def call_batch_norm(batch_norm, chan_inp):
    return batch_norm(chan_inp)

class CNN(nn.Module):
    def __init__(self, input_channel=3, n_outputs=10, dropout_rate=0.25, momentum=0.1):
        self.dropout_rate = dropout_rate
        self.momentum = momentum
        super(CNN, self).__init__()
        self.conv_lay1 = nn.Conv2d(input_channel, 64, kernel_size=3, stride=1, padding=1)
        # self.conv_lay2=nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv_lay3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        # self.conv_lay4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv_lay5 = nn.Conv2d(128, 196, kernel_size=3, stride=1, padding=1)
        self.conv_lay6 = nn.Conv2d(196, 16, kernel_size=3, stride=1, padding=1)
        self.bch_norm_lay1 = nn.BatchNorm2d(64, momentum=self.momentum)
        # self.bch_norm_lay2 = nn.BatchNorm2d(64, momentum=self.momentum)
        self.bch_norm_lay3 = nn.BatchNorm2d(128, momentum=self.momentum)
        # self.bch_norm_lay4 = nn.BatchNorm2d(128, momentum=self.momentum)
        self.bch_norm_lay5 = nn.BatchNorm2d(196, momentum=self.momentum)
        self.bch_norm_lay6 = nn.BatchNorm2d(16, momentum=self.momentum)
        self.lin_lay1 = nn.Linear(256, n_outputs)
    def forward(self, x):
        # inp = x
        inp = self.c1(x)
        inp = F.relu(call_batch_norm(self.conv_lay1, inp))
        # inp = self.conv_lay2(inp)
        # inp = F.relu(call_batch_norm(self.bch_norm_lay2, inp))
        inp = F.max_pool2d(inp, kernel_size=2, stride=2)

        inp = self.conv_lay3(inp)
        inp = F.relu(call_batch_norm(self.bch_norm_lay3, inp))
        # inp = self.conv_lay4(inp)
        # inp = F.relu(call_batch_norm(self.bch_norm_lay4, inp))
        inp = F.max_pool2d(inp, kernel_size=2, stride=2)

        inp = self.conv_lay5(inp)
        inp = F.relu(call_batch_norm(self.bch_norm_lay5, inp))
        inp = self.conv_lay6(inp)
        inp = F.relu(call_batch_norm(self.bch_norm_lay6, inp))
        inp = F.max_pool2d(inp, kernel_size=2, stride=2)

        inp = inp.view(inp.size(0), -1)
        logit = self.lin_lay1(inp)
        return logit


class MLPNet(nn.Module):
    def __init__(self):
        super(MLPNet, self).__init__()
        self.full_conn1 = nn.Linear(28*28, 256)
        self.full_conn2 = nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.full_conn1(x))
        x = self.full_conn2(x)
        return x

# If one wants to freeze layers of a network
# for param in model.parameters():
#   param.requires_grad = False


T_START = time.time()

"""
Configuration
"""

DATASET = ARGS.dataset
NOISE_RATE = ARGS.noise_rate
NOISE_TYPE = ARGS.noise_type
DATA_AUG = bool(ARGS.data_aug)
BATCH_SIZE = ARGS.batch_size
LOSS_NAME = ARGS.loss_name
NUM_EPOCH = ARGS.num_epoch
NUM_RUNS = ARGS.num_runs

LEARNING_RATE = 2e-4
MODE = "batch_rewgt"

"""
Loss Function
"""


if LOSS_NAME == "cce":
    loss_fn = WeightedCCE(k=1, reduction="none")
else:
    raise NotImplementedError("Loss Function Not Implemented.\n")
print("\n==============\nloss_name: {}\n=============\n".format(LOSS_NAME))


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


for run in range(NUM_RUNS):

    T_START = time.time()

    epoch_loss_train = []
    epoch_acc_train = []
    epoch_loss_test = []
    epoch_acc_test = []

    best_acc_val = 0.

    chkpt_path = f"./checkpoint/{MODE}/{DATASET}/\
                {NOISE_TYPE}/0{str(int(NOISE_RATE*10))}/run_{str(run)}/"

    res_path = f"./results_pkl/{MODE}/{DATASET}/\
                {NOISE_TYPE}/0{str(int(NOISE_RATE*10))}/run_{str(run)}/"

    plt_path = f"./plots/{MODE}/{DATASET}/\
                {NOISE_TYPE}/0{str(int(NOISE_RATE*10))}/run_{str(run)}/"

    log_dirs_path = f"./runs/{MODE}/{DATASET}/\
                {NOISE_TYPE}/0{str(int(NOISE_RATE*10))}/run_{str(run)}/"

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
            except OSError as err:
                print(f"Error: {f} : {err.strerror}")
        print("\nLog files cleared...\n")

    print("\n============ PATHS =================\n")
    print(f"chkpt_path: {chkpt_path}")
    print(f"res_path: {res_path}")
    print(f"plt_path: {plt_path}")
    print(f"log_dirs_path: {log_dirs_path}")
    print(f"file name: {MODE}-aug-{DATA_AUG}-{DATASET}-{LOSS_NAME}-\
            {NOISE_TYPE}-nr-0{str(int(NOISE_RATE * 10))}-run-{str(run)}.pt")
    print("\n=============================\n")


    """
    Training/Validation/Test Data
    """

    dat, ids = read_data(NOISE_TYPE, NOISE_RATE, DATASET, data_aug=DATA_AUG, mode=MODE)

    X_temp, y_temp, X_train, y_train = dat[0], dat[1], dat[2], dat[3]
    X_val, y_val, X_test, y_test = dat[4], dat[5], dat[6], dat[7]
    idx_tot, idx_train, idx_val = ids[0], ids[1], ids[2]

    if NOISE_RATE > 0.:
        idx_tr_clean_ref, idx_tr_noisy_ref = ids[3], ids[4]


    print("\n=============================\n")
    print("X_train: ", X_train.shape, " y_train: ", y_train.shape, "\n")
    print("X_val: ", X_val.shape, " y_val: ", y_val.shape, "\n")
    print("X_test: ", X_test.shape, " y_test: ", y_test.shape, "\n")
    print("\n=============================\n")
    print("\n Noise Type: {}, Noise Rate: {} \n".format(NOISE_TYPE, NOISE_RATE))

    """
    Create Dataset Loader
    """

    ### Train. set
    # .as_tensor() avoids copying, .Tensor() creates a new copy
    tensor_x_train = torch.Tensor(X_train)
    tensor_y_train = torch.Tensor(y_train)
    tensor_id_train = torch.from_numpy(np.asarray(list(range(X_train.shape[0]))))

    dataset_train = torch.utils.data.TensorDataset(tensor_x_train, tensor_y_train, tensor_id_train)
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)

    # Val. set
    tensor_x_val = torch.Tensor(X_val)
    tensor_y_val = torch.Tensor(y_val)
    # tensor_id_val = torch.Tensor(idx_val)

    VAL_SIZE = 1000
    dataset_val = torch.utils.data.TensorDataset(tensor_x_val, tensor_y_val) #, tensor_id_val)
    val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=VAL_SIZE, shuffle=True)

    # Test set
    tensor_x_test = torch.Tensor(X_test)
    tensor_y_test = torch.Tensor(y_test)

    TEST_SIZE = 1000
    dataset_test = torch.utils.data.TensorDataset(tensor_x_test, tensor_y_test)
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=TEST_SIZE, shuffle=True)


    """
    Initialize n/w and optimizer
    """

    if DATASET == "mnist":
        model = MNIST_NN()
        # model = MLPNet()
    elif DATASET == "cifar10":
        model = CNN()

        # Adjust learning rate and betas for Adam Optimizer
        mom1 = 0.9
        mom2 = 0.1
        alpha_plan = [LEARNING_RATE] * NUM_EPOCH
        beta1_plan = [mom1] * NUM_EPOCH
        for i in range(10, NUM_EPOCH):
            alpha_plan[i] = float(NUM_EPOCH - i) / (NUM_EPOCH - 10) * LEARNING_RATE
            beta1_plan[i] = mom2

        def adjust_learning_rate(opt, num_epoch):
            for param_group in opt.param_groups:
                param_group['lr'] = alpha_plan[num_epoch]
                param_group['betas'] = (beta1_plan[num_epoch], 0.999)

    # params = list(model.parameters())
    model = model.to(DEVICE)
    print(model)

    """
    Optimizer and LR Scheduler

    Multiple LR Schedulers: https://github.com/pytorch/pytorch/pull/26423
    """
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
    lr_scheduler_1 = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                    factor=0.1, patience=5,
                                                    verbose=True, threshold=0.0001,
                                                    threshold_mode='rel', cooldown=0,
                                                    min_lr=1e-5, eps=EPS
                                                    )
    # lr_scheduler_2 = lr_scheduler.MultiStepLR(optimizer, milestones=[30, 80], gamma=0.1)
    lr_scheduler_2 = lr_scheduler.MultiStepLR(optimizer, milestones=[40, 80], gamma=0.1)
    ## optimizer = optim.RMSprop(model.parameters(), lr=0.0001)
    lr_scheduler_3 = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)


    """
    Setting up Tensorbard
    """
    writer = SummaryWriter(log_dirs_path)
    # writer.add_graph(model, (tensor_x_train[0].unsqueeze(1)).to(DEVICE))
    # writer.close()

    epoch_idx_sel_tr_agg = np.zeros((len(train_loader.dataset), NUM_EPOCH))

    for epoch in range(NUM_EPOCH):

        #Training set performance
        loss_train, acc_train, idx_sel_tr_agg = batch_rewgt_train(train_loader, model)
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

        if DATASET in ["mnist", "svhn"]:
            lr_scheduler_1.step(loss_val)
        elif DATASET in ["cifar10", "cifar100"]:
            lr_scheduler_1.step(loss_val)
            # lr_scheduler_2.step()
            # adjust_learning_rate(optimizer, epoch)
        elif DATASET == "news":
            lr_scheduler_1.step(loss_val)
            # lr_scheduler_2.step()

        epoch_loss_train.append(loss_train)
        epoch_acc_train.append(acc_train)
        epoch_loss_test.append(loss_test)
        epoch_acc_test.append(acc_test)

        epoch_idx_sel_tr_agg[:, epoch] = idx_sel_tr_agg

        # Update best_acc_val
        if epoch == 0:
            best_acc_val = acc_val

        if acc_val > best_acc_val:
            best_acc_val = acc_val
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), chkpt_path + f"{MODE}-{DATASET}-{NOISE_TYPE}-\
                                                        {LOSS_NAME}-nr-\
                                                        0{str(int(NOISE_RATE * 10))}-\
                                                        mdl-wts-run-{str(run)}.pt")
            print("Model weights updated...\n")

        print(f"Epoch: {epoch}, lr: {optimizer.param_groups[0]['lr']}, loss_train: {loss_train},\
                loss_val: {loss_val}, loss_test: {loss_test}, acc_train: {acc_train},\
                 acc_val: {acc_val}, acc_test: {acc_test}\n")


    # Test accuracy on the best_val MODEL
    loss_test, acc_test = test(test_loader, model, run, use_best=False)
    print("Test set performance - test_acc: {}, test_loss: {}\n".format(acc_test, loss_test))

    # Print the elapsed time
    elapsed = time.time() - T_START
    print("\n elapsed time: \n", elapsed)

    """
    Save results
    """
    with open(res_path+ f"{MODE}-{DATASET}-{LOSS_NAME}-{NOISE_TYPE}-nr-\
                0{str(int(NOISE_RATE * 10))}-run-{str(run)}.pickle", 'wb') as f:
        if NOISE_RATE > 0.:
            pickle.dump({'epoch_loss_train': np.asarray(epoch_loss_train),
                         'epoch_acc_train': np.asarray(epoch_acc_train),
                         'epoch_loss_test': np.asarray(epoch_loss_test),
                         'epoch_acc_test': np.asarray(epoch_acc_test),
                         'y_train_org': y_temp[idx_train], 'y_train':y_train,
                         'epoch_idx_sel_tr_agg': epoch_idx_sel_tr_agg,
                         'idx_tr_clean_ref': idx_tr_clean_ref,
                         'idx_tr_noisy_ref': idx_tr_noisy_ref,
                         'num_epoch': NUM_EPOCH, 'time_elapsed': elapsed
                         }, f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            pickle.dump({'epoch_loss_train': np.asarray(epoch_loss_train),
                         'epoch_acc_train': np.asarray(epoch_acc_train),
                         'epoch_loss_test': np.asarray(epoch_loss_test),
                         'epoch_acc_test': np.asarray(epoch_acc_test),
                         'y_train_org': y_temp[idx_train], 'y_train':y_train,
                         'num_epoch': NUM_EPOCH,
                         'time_elapsed': elapsed
                         }, f, protocol=pickle.HIGHEST_PROTOCOL)
        print("Pickle file saved: " + res_path+ f"{MODE}-{DATASET}-{LOSS_NAME}-{NOISE_TYPE}-\
                                                nr-0{str(int(NOISE_RATE * 10))}-\
                                                run-{str(run)}.pickle\n")
