from __future__ import print_function, absolute_import

import os
import time
import pickle
import pathlib
from tqdm import tqdm
import copy
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter

from data import read_data
from losses import get_loss

# set seed for reproducibility
torch.manual_seed(1337)
np.random.seed(3459)
# tf.set_random_seed(3459)

torch.autograd.set_detect_anomaly(True)

eps = 1e-8


parser = argparse.ArgumentParser(description='PyTorch MNIST Risk Min. Training')
parser.add_argument('-dat', '--dataset', default="mnist",type=str, help="dataset")
parser.add_argument('-nr', '--noise_rate', default=0.4, type=float, help="noise rate")
parser.add_argument('-nt', '--noise_type', default="sym", type=str, help="noise type")
parser.add_argument('-loss', '--loss_name', default="cce",type=str, help="loss name")
parser.add_argument('-da', '--data_aug', default=0, type=int, help="data augmentation (0 or 1)")
parser.add_argument("-bs", "--batch_size", default=128, type=int, help="batch size")
parser.add_argument('-ep', '--num_epoch', default=100, type=int, help="number of epochs")
parser.add_argument('-k', '--k_sel', default=1, type=float, help="k-th quantile")
parser.add_argument('-run', '--num_runs', default=1, type=int, help="number of runs/simulations")
args = parser.parse_args()


def accuracy(true_label, pred_label):
	num_samples = true_label.shape[0]
	err = [1 if (pred_label[i] != true_label[i]).sum()==0 else 0 for i in range(num_samples)]
	acc = 1 - (sum(err)/num_samples)
	return acc


def risk_min_train(dataset_train, train_loader, model):

    loss_train = 0.
    acc_train = 0.
    correct = 0

    model.train()

    for batch_id, (x, y, idx) in tqdm(enumerate(train_loader)):

        y = y.type(torch.LongTensor)
        
        # Transfer data to the GPU
        x, y, idx = x.to(device), y.to(device), idx.to(device)

        output = model(x)
        pred_prob = F.softmax(output, dim=1)
        pred = torch.argmax(pred_prob, dim=1)
        pred_prob = torch.clamp(pred_prob, eps, 1-eps)
        loss_batch = loss_fn(output, y)

        optimizer.zero_grad()
        loss_batch.mean().backward()
        optimizer.step()

        loss_train += torch.mean(loss_batch).item()
        correct += (pred.eq(y.to(device))).sum().item()

        batch_cnt = batch_id + 1

    loss_train /= batch_cnt
    acc_train = 100.*correct/len(train_loader.dataset)

    return loss_train, acc_train

def test(data_loader, model, run, use_best=False):

    loss_test = 0.
    correct = 0

    model.eval()

    with torch.no_grad():
        for batch_id, (x, y) in enumerate(data_loader):
            if use_best == True:
                # load best model weights
                model.load_state_dict(torch.load(chkpt_path + "%s-%s-%s-%s-nr-0%s-mdl-wts-\
                                                 run-%s.pt" % (mode, dataset, loss_name,
                                                               noise_type,
                                                               str(int(noise_rate* 10)),
                                                               str(run))))
                model = model.to(device)

            y = y.type(torch.LongTensor)      
            x, y = x.to(device), y.to(device)

            output = model(x)
            pred_prob = F.softmax(output, dim=1)
            pred = torch.argmax(pred_prob, dim=1)

            loss_batch = loss_fn(output, y)
            loss_test += torch.mean(loss_batch).item()
            correct += (pred.eq(y.to(device))).sum().item()

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
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x)) / self.temp
        return x

    def num_flat_features(self, x):
        size = x.size()[1:] # all dims except batch_size dim
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

def call_bn(bn, x):
    return bn(x)

class CNN(nn.Module):
    def __init__(self, input_channel=3, n_outputs=10, dropout_rate=0.25, momentum=0.1):
        self.dropout_rate = dropout_rate
        self.momentum = momentum 
        super(CNN, self).__init__()
        self.c1=nn.Conv2d(input_channel, 64,kernel_size=3,stride=1, padding=1)        
        # self.c2=nn.Conv2d(64,64,kernel_size=3,stride=1, padding=1)        
        self.c3=nn.Conv2d(64,128,kernel_size=3,stride=1, padding=1)        
        # self.c4=nn.Conv2d(128,128,kernel_size=3,stride=1, padding=1)        
        self.c5=nn.Conv2d(128,196,kernel_size=3,stride=1, padding=1)        
        self.c6=nn.Conv2d(196,16,kernel_size=3,stride=1, padding=1)        
        self.linear1=nn.Linear(256, n_outputs)
        self.bn1=nn.BatchNorm2d(64, momentum=self.momentum)
        # self.bn2=nn.BatchNorm2d(64, momentum=self.momentum)
        self.bn3=nn.BatchNorm2d(128, momentum=self.momentum)
        # self.bn4=nn.BatchNorm2d(128, momentum=self.momentum)
        self.bn5=nn.BatchNorm2d(196, momentum=self.momentum)
        self.bn6=nn.BatchNorm2d(16, momentum=self.momentum)

    def forward(self, x,):
        h=x
        h=self.c1(h)
        h=F.relu(call_bn(self.bn1, h))
        # h=self.c2(h)
        # h=F.relu(call_bn(self.bn2, h))
        h=F.max_pool2d(h, kernel_size=2, stride=2)

        h=self.c3(h)
        h=F.relu(call_bn(self.bn3, h))
        # h=self.c4(h)
        # h=F.relu(call_bn(self.bn4, h))
        h=F.max_pool2d(h, kernel_size=2, stride=2)

        h=self.c5(h)
        h=F.relu(call_bn(self.bn5, h))
        h=self.c6(h)
        h=F.relu(call_bn(self.bn6, h))
        h=F.max_pool2d(h, kernel_size=2, stride=2)

        h = h.view(h.size(0), -1)
        logit=self.linear1(h)
        return logit

class MLPNet(nn.Module):
    def __init__(self):
        super(MLPNet, self).__init__()
        self.fc1 = nn.Linear(28*28, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# If one wants to freeze layers of a network
# for param in model.parameters():
#   param.requires_grad = False


t_start = time.time()

"""
Configuration
"""

random_state = 422

dataset = args.dataset
noise_rate = args.noise_rate
noise_type = args.noise_type
data_aug = bool(args.data_aug)
batch_size = args.batch_size
loss_name = args.loss_name
num_epoch = args.num_epoch
num_runs = args.num_runs

batch_size = 128
learning_rate = 2e-4

mode = "risk_min" 

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


for run in range(num_runs):

    t_start = time.time()

    epoch_loss_train = []
    epoch_acc_train = []
    epoch_loss_test = []
    epoch_acc_test = []

    best_acc_val = 0.

    chkpt_path = f"./checkpoint/{mode}/{dataset}/{noise_type}/0{str(int(noise_rate*10))}/run_{str(run)}/"

    res_path = f"./results_pkl/{mode}/{dataset}/{noise_type}/0{str(int(noise_rate*10))}/run_{str(run)}/"

    plt_path = f"./plots/{mode}/{dataset}/{noise_type}/0{str(int(noise_rate*10))}/run_{str(run)}/"

    log_dirs_path = f"./runs/{mode}/{dataset}/{noise_type}/0{str(int(noise_rate*10))}/run_{str(run)}/"

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
    print("file name: %s-aug-%s-%s-%s-%s-nr-0%s-run-%s.pt" % (mode, data_aug, dataset,
                                                              loss_name, noise_type,
                                                              str(int(noise_rate * 10)),
                                                              str(run)))
    print("\n=============================\n")


    """
    Training/Validation/Test Data
    """

    if dataset in ["mnist", "cifar10"]:
        dat, ids = read_data(noise_type, noise_rate, dataset,
                             data_aug=data_aug, mode=mode)

        X_temp, y_temp = dat[0], dat[1]
        X_train, y_train = dat[2], dat[3]
        X_val, y_val = dat[4], dat[5]
        X_test, y_test = dat[6], dat[7]

        if noise_rate > 0.:
            idx, idx_train, idx_val = ids[0], ids[1], ids[2]
            idx_tr_clean_ref, idx_tr_noisy_ref = ids[3], ids[4]
            idx_train_clean, idx_train_noisy = ids[5], ids[6]
        
        else:
            idx, idx_train, idx_val = ids[0], ids[1], ids[2]
            idx_train_clean, idx_train_noisy = ids[3], ids[4]

        if int(np.min(y_temp)) == 0:
            num_class = int(np.max(y_temp) + 1)
        else:
            num_class = int(np.max(y_temp))
    else:
        raise SystemExit("Dataset not supported.\n")

    print("\n=============================\n")
    print("X_train: ", X_train.shape, " y_train: ", y_train.shape, "\n")
    print("X_val: ", X_val.shape, " y_val: ", y_val.shape, "\n")
    print("X_test: ", X_test.shape, " y_test: ", y_test.shape, "\n")    
    print("\n=============================\n")

    print("\n Noise Type: {}, Noise Rate: {} \n".format(noise_type, noise_rate))


    """
    Create Dataset Loader
    """

    # Train. set
    tensor_x_train = torch.Tensor(X_train) # .as_tensor() avoids copying, .Tensor() creates a new copy
    tensor_y_train = torch.Tensor(y_train) # .as_tensor() avoids copying, .Tensor() creates a new copy
    tensor_id_train = torch.from_numpy(np.asarray(list(range(X_train.shape[0]))))

    dataset_train = torch.utils.data.TensorDataset(tensor_x_train, tensor_y_train, tensor_id_train)
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True)

    # Val. set
    tensor_x_val = torch.Tensor(X_val)
    tensor_y_val = torch.Tensor(y_val)
    # tensor_id_val = torch.Tensor(idx_val)

    val_size = 1000
    dataset_val = torch.utils.data.TensorDataset(tensor_x_val, tensor_y_val) #, tensor_id_val)
    val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=val_size, shuffle=True)

    # Test set
    tensor_x_test = torch.Tensor(X_test)
    tensor_y_test = torch.Tensor(y_test)

    test_size = 1000
    dataset_test = torch.utils.data.TensorDataset(tensor_x_test, tensor_y_test)
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=test_size, shuffle=True)

    """
    Initialize n/w and optimizer
    """

    if dataset == "mnist":
        model = MLPNet()
    elif dataset == "cifar10":
        model = CNN()
        # learning_rate = 1e-3

        # Adjust learning rate and betas for Adam Optimizer
        mom1 = 0.9
        mom2 = 0.1
        alpha_plan = [learning_rate] * num_epoch
        beta1_plan = [mom1] * num_epoch
        for i in range(10, num_epoch):
            alpha_plan[i] = float(num_epoch - i) / (num_epoch - 10) * learning_rate
            beta1_plan[i] = mom2

        def adjust_learning_rate(optimizer, epoch):
            for param_group in optimizer.param_groups:
                param_group['lr']=alpha_plan[epoch]
                param_group['betas']=(beta1_plan[epoch], 0.999)
    else:
        raise SystemExit("Dataset not supported.\n")
    model = model.to(device)
    print(model)

    """
    Loss Function
    """

    print("\n===========\nloss: {}\n===========\n".format(loss_name))

    kwargs = {}

    if loss_name == "rll":
        kwargs['alpha'] = 0.1 # 0.45 # 0.01
    elif loss_name == "gce":
        kwargs['q'] = 0.7
    elif loss_name == "norm_mse":
        kwargs['alpha'] = 0.1
        kwargs['beta'] = 1.

    loss_fn = get_loss(loss_name, num_class, reduction="none", **kwargs)

    """
    Optimizer and LR Scheduler

    Multiple LR Schedulers: https://github.com/pytorch/pytorch/pull/26423
    """
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    lr_scheduler_1 = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                    factor=0.1, patience=5, verbose=True, threshold=0.0001,
                    threshold_mode='rel', cooldown=0, min_lr=1e-5, eps=1e-08)
    lr_scheduler_2 = lr_scheduler.MultiStepLR(optimizer, milestones=[40,80], gamma=0.1)
    lr_scheduler_3 = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    """
    Setting up Tensorbard
    """
    writer = SummaryWriter(log_dirs_path)
    # writer.add_graph(model, (tensor_x_train[0].unsqueeze(1)).to(device))
    # writer.close()

    for epoch in range(num_epoch):

        #Training set performance
        loss_train, acc_train = risk_min_train(dataset_train, train_loader, model)
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

        # Learning Rate Scheduler Update
        if dataset == "mnist":
            lr_scheduler_1.step(loss_val)
        elif dataset == "cifar10":
            lr_scheduler_1.step(loss_val)
            # lr_scheduler_2.step()
            # adjust_learning_rate(optimizer, epoch)

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
            torch.save(model.state_dict(), chkpt_path + "%s-%s-%s-%s-nr-0%s-mdl-wts-\
                       run-%s.pt" % (mode, dataset, noise_type, loss_name,
                                     str(int(noise_rate * 10)), str(run)))
            print("Model weights updated...\n")

        print("Epoch: {}, lr: {}, loss_train: {}, loss_val: {}, loss_test: {:.3f}, acc_train: {},\
              acc_val: {}, acc_test: {:.3f}\n".format(epoch, optimizer.param_groups[0]['lr'],
                                                      loss_train, loss_val, loss_test,
                                                      acc_train, acc_val, acc_test))


    # Test accuracy on the best_val MODEL
    loss_test, acc_test = test(test_loader, model, run, use_best=False)
    print("Test set performance - test_acc: {}, test_loss: \
          {}\n".format(acc_test, loss_test))

    # Print the elapsed time
    elapsed = time.time() - t_start
    print("\nelapsed time: \n", elapsed)

    """
    Save results
    """
    with open(res_path + "%s-%s-%s-%s-nr-0%s-run-%s.pickle" % (mode, dataset, loss_name,
                                                               noise_type, str(int(noise_rate
                                                               * 10)), str(run)), 'wb') as f:
        pickle.dump({'epoch_loss_train': np.asarray(epoch_loss_train), 
                     'epoch_acc_train': np.asarray(epoch_acc_train), 
                     'epoch_loss_test': np.asarray(epoch_loss_test), 
                     'epoch_acc_test': np.asarray(epoch_acc_test), 
                     'y_train_org': y_temp[idx_train], 'y_train':y_train,
                     'num_epoch': num_epoch,
                     'time_elapsed': elapsed}, f, protocol=pickle.HIGHEST_PROTOCOL)
        print("Pickle file saved: " + res_path+ "%s-%s-%s-%s-nr-0%s-\
              run-%s.pickle" % (mode, dataset, loss_name, noise_type,
                                str(int(noise_rate * 10)), str(run)), "\n")