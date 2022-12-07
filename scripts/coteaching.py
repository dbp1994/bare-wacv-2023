from __future__ import print_function, absolute_import

import os
import time
import pickle
import copy
import argparse
import pathlib
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter

from data import read_data
from losses import get_loss

# set seed for reproducibility
# torch.manual_seed(1337)
# np.random.seed(3459)
torch.manual_seed(333)
np.random.seed(4578)
# tf.set_random_seed(3459)

torch.autograd.set_detect_anomaly(True)

EPS = 1e-8

# Loss functions
def loss_coteaching(y_1, y_2, t, forget_rate, ind, noise_or_not, loss_fn):
    """
    Adapted from:
    https://github.com/xingruiyu/coteaching_plus/blob/master/loss.py
    """
    loss_1 = loss_fn(y_1, t)
    _, ind_1_sorted = torch.sort(loss_1)
    # loss_1_sorted = loss_1[ind_1_sorted.tolist()]

    loss_2 = loss_fn(y_2, t)
    _, ind_2_sorted = torch.sort(loss_2)
    # loss_2_sorted = loss_2[ind_2_sorted.tolist()]

    remember_rate = 1 - forget_rate
    num_remember = int(remember_rate * len(ind_1_sorted.tolist()))

    ind_1_update=ind_1_sorted.tolist()[:num_remember]
    ind_2_update=ind_2_sorted.tolist()[:num_remember]
    if len(ind_1_update) == 0 or len(ind_2_update) == 0:
        ind_1_update = ind_1_sorted.tolist()
        ind_2_update = ind_2_sorted.tolist()
        num_remember = len(ind_1_update)

    pure_ratio_1 = np.sum(noise_or_not[ind[ind_1_update].tolist()])/float(num_remember)
    pure_ratio_2 = np.sum(noise_or_not[ind[ind_2_update].tolist()])/float(num_remember)

    loss_1_update = loss_fn(y_1[ind_2_update], t[ind_2_update])
    loss_2_update = loss_fn(y_2[ind_1_update], t[ind_1_update])

    return loss_1_update, loss_2_update, pure_ratio_1, pure_ratio_2, ind_1_update, ind_2_update

def loss_coteaching_plus(logits, logits2, labels, forget_rate, ind, noise_or_not, step, loss_fn):
    """
    Adapted from:
    https://github.com/xingruiyu/coteaching_plus/blob/master/loss.py
    """
    outputs = F.softmax(logits, dim=1)
    outputs2 = F.softmax(logits2, dim=1)

    pred1 = torch.argmax(logits, dim=1)
    pred2 = torch.argmax(logits2, dim=1)

    pred1, pred2 = pred1.cpu().numpy(), pred2.cpu().numpy()

    logical_disagree_id=np.zeros(labels.size(), dtype=bool)
    disagree_id = []
    for idx, p1 in enumerate(pred1): 
        if p1 != pred2[idx]:
            disagree_id.append(idx) 
            logical_disagree_id[idx] = True
    
    temp_disagree = np.asarray(ind.tolist())*logical_disagree_id.astype(np.int64)
    ind_disagree = np.asarray([i for i in temp_disagree if i != 0]).transpose()
    try:
        assert ind_disagree.shape[0]==len(disagree_id)
    except:
        disagree_id = disagree_id[:ind_disagree.shape[0]]
     
    _update_step = np.logical_or(logical_disagree_id, step < 5000).astype(np.float32)
    update_step = torch.from_numpy(_update_step).cuda()

    if len(disagree_id) > 0:
        update_labels = labels[disagree_id]
        update_outputs = outputs[disagree_id] 
        update_outputs2 = outputs2[disagree_id]
        
        loss_1, loss_2, pure_ratio_1, pure_ratio_2, idx_sel_1, idx_sel_2 = loss_coteaching(update_outputs, 
                                                    update_outputs2, update_labels, 
                                                    forget_rate, ind_disagree, 
                                                    noise_or_not, loss_fn)
    else:
        update_labels = labels
        update_outputs = outputs
        update_outputs2 = outputs2

        loss_1_tmp = loss_fn(update_outputs, update_labels)
        loss_2_tmp = loss_fn(update_outputs2, update_labels)

        loss_1 = torch.sum(update_step*loss_1_tmp)/labels.size()[0]
        loss_2 = torch.sum(update_step*loss_2_tmp)/labels.size()[0]
 
        pure_ratio_1 = np.sum(noise_or_not[ind.tolist()])/ind.shape[0]
        pure_ratio_2 = np.sum(noise_or_not[ind.tolist()])/ind.shape[0]

        idx_sel_1 = list(range(logits.shape[0]))
        idx_sel_2 = list(range(logits.shape[0]))
        
    return loss_1, loss_2, pure_ratio_1, pure_ratio_2, idx_sel_1, idx_sel_2


class CIFAR_NN(nn.Module):
    def __init__(self, temp=1.0):
        super(CIFAR_NN, self).__init__()

        # 1 I/P channel, 6 O/P channels, 5x5 conv. kernel
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5)

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(480, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        self.temp = temp
    
    def forward(self, x):

        # x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2), stride=2)
        # x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2), stride=2)
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x)) / self.temp
        return x

    def num_flat_features(self, x):
        size = x.size()[1:] # all dims except batch_size dim
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class CNN_small(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN_small, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

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

def coteaching_train(epoch, train_loader, model_1, model_2, rate_schedule, noise_or_not, loss_fn):

    loss_train_1 = 0.
    acc_train_1 = 0.
    correct_1 = 0

    loss_train_2 = 0.
    acc_train_2 = 0.
    correct_2 = 0

    model_1.train()
    model_2.train()

    loss_train_agg_1 = np.zeros((len(train_loader.dataset), ))
    acc_train_agg_1 = np.zeros((len(train_loader.dataset), ))
    pred_train_agg_1 = np.zeros((len(train_loader.dataset), ))

    loss_train_agg_2 = np.zeros((len(train_loader.dataset), ))
    acc_train_agg_2 = np.zeros((len(train_loader.dataset), ))
    pred_train_agg_2 = np.zeros((len(train_loader.dataset), ))

    idx_sel_tr_agg_1 = np.zeros((len(train_loader.dataset), ))
    idx_sel_tr_agg_2 = np.zeros((len(train_loader.dataset), ))

    lab_prec_agg_1 = 0.
    lab_prec_agg_2 = 0.

    for batch_id, (x, y, idx) in tqdm(enumerate(train_loader)):


        y = y.type(torch.LongTensor)

        if dataset == "news":
            x = x.type(torch.LongTensor)

        x, y, idx = x.to(device), y.to(device), idx.to(device)

        output_1 = model_1(x)
        pred_prob_1 = F.softmax(output_1, dim=1)
        pred_1 = torch.argmax(pred_prob_1, dim=1)

        output_2 = model_2(x)
        pred_prob_2 = F.softmax(output_2, dim=1)
        pred_2 = torch.argmax(pred_prob_2, dim=1)

        if mode == "coteaching":
            loss_batch_1, loss_batch_2, lab_prec_1,
            lab_prec_2, idx_sel_1, idx_sel_2 = loss_coteaching(output_1, output_2,
                                                               y, rate_schedule,
                                                               idx, noise_or_not,
                                                               loss_fn)
        elif mode == "coteaching_plus":
            if epoch < init_epoch:
                loss_batch_1, loss_batch_2, lab_prec_1,
                lab_prec_2, idx_sel_1, idx_sel_2 = loss_coteaching(output_1, output_2,
                                                                   y, rate_schedule,
                                                                   idx, noise_or_not, 
                                                                   loss_fn)
            else:
                loss_batch_1, loss_batch_2, lab_prec_1,
                lab_prec_2, idx_sel_1, idx_sel_2 = loss_coteaching_plus(output_1, output_2,
                                                                        y, rate_schedule,
                                                                        idx, noise_or_not,
                                                                        ((epoch-1)*bs_step)
                                                                        +batch_id+1, loss_fn)

        optimizer_1.zero_grad()
        loss_batch_1.mean().backward()
        optimizer_1.step()

        optimizer_2.zero_grad()
        loss_batch_2.mean().backward()
        optimizer_2.step()

        loss_train_1 += torch.mean(loss_batch_1).item()
        correct_1 += (pred_1.eq(y)).sum().item()

        loss_train_2 += torch.mean(loss_batch_2).item()
        correct_2 += (pred_2.eq(y)).sum().item()

        lab_prec_agg_1 += lab_prec_1
        lab_prec_agg_2 += lab_prec_2

        batch_cnt = batch_id + 1

        loss_train_agg_1[list(map(int, idx_sel_1))] = np.asarray(loss_batch_1.tolist())
        acc_train_agg_1[list(map(int, idx.tolist()))] = np.asarray((pred_1.eq(y)).tolist())
        pred_train_agg_1[list(map(int, idx.tolist()))] = np.asarray(pred_1.tolist())

        loss_train_agg_2[list(map(int, idx_sel_2))] = np.asarray(loss_batch_2.tolist())
        acc_train_agg_2[list(map(int, idx.tolist()))] = np.asarray((pred_2.eq(y)).tolist())
        pred_train_agg_2[list(map(int, idx.tolist()))] = np.asarray(pred_2.tolist())

        idx_tmp = np.zeros((x.shape[0], ))
        idx_tmp[idx_sel_1] = 1.
        idx_sel_tr_agg_1[list(map(int, idx.tolist()))] = np.asarray(idx_tmp)

        idx_tmp = np.zeros((x.shape[0], ))
        idx_tmp[idx_sel_2] = 1.
        idx_sel_tr_agg_2[list(map(int, idx.tolist()))] = np.asarray(idx_tmp)


    loss_train_1 /= batch_cnt
    acc_train_1 = 100.*correct_1/len(train_loader.dataset)

    loss_train_2 /= batch_cnt
    acc_train_2 = 100.*correct_2/len(train_loader.dataset)

    lab_prec_agg_1 /= batch_cnt
    lab_prec_agg_2 /= batch_cnt


    return (loss_train_1, acc_train_1, loss_train_agg_1, acc_train_agg_1, pred_train_agg_1, 
            loss_train_2, acc_train_2, loss_train_agg_2, acc_train_agg_2, pred_train_agg_2, 
            lab_prec_agg_1, lab_prec_agg_2, idx_sel_tr_agg_1, idx_sel_tr_agg_2)


def test(data_loader, model, run, use_best=False):

    loss_test = 0.
    acc_test = 0.
    correct = 0

    model.eval()

    print("Testing...\n")

    with torch.no_grad():
        for batch_id, (x, y) in enumerate(data_loader):
            if use_best == True:
                # load best model weights
                model.load_state_dict(torch.load(chkpt_path + 
                                                 f"{mode}-bn-{batch_norm}-wd-\
                                                 {str(weight_decay)}-aug-\
                                                 {data_aug}-{arch}-{dataset}-\
                                                 {loss_name}-{noise_type}-nr-\
                                                 0{str(int(noise_rate * 10))}-\
                                                 run-{str(run)}.pt")['model_state_dict'])
                model = model.to(device)

            y = y.type(torch.LongTensor)

            if dataset == "news":
                x = x.type(torch.LongTensor)
        
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


def rate_schedule(arg, num_gradual=10,exponent=1):

    """
    arguments:

            num_gradual - how many epochs for linear drop rate, can be 5, 10, 15. 
                        This parameter is equal to Tk for R(T) in Co-teaching
            exponent - exponent of the forget rate, can be 0.5, 1, 2. 
                        This parameter is equal to c in Tc for R(T) in Co-teaching

    """

    rate = np.ones(arg.num_epoch)*arg.noise_rate
    rate[:num_gradual] = np.linspace(0, arg.noise_rate**exponent, num_gradual)
    return rate


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="PyTorch 'Co-Teaching' Training")
    parser.add_argument("-dat", "--dataset", default="cifar10", type=str, help="dataset")
    parser.add_argument("-mod", "--mode", default="coteaching", type=str, help="mode: [coteaching, coteaching_plus]")
    parser.add_argument("-nr","--noise_rate", default=0.4, type=float, help="noise_rate")
    parser.add_argument("-nt","--noise_type", default="sym", type=str, help="noise_type")
    parser.add_argument("-loss","--loss_name", default="cce", type=str, help="loss_name")
    parser.add_argument("-arch", "--architecture", default="cnn", type=str, help="architecture")
    parser.add_argument("-bn", "--batch_norm", default=1, type=int, help="Batch Normalization", choices=[0, 1])
    parser.add_argument("-wd", "--weight_decay", default=0., type=float, help="weight decay for optimizer")
    parser.add_argument("-da", "--data_aug", default=1, type=int, help="data augmentation", choices=[0, 1])
    parser.add_argument("-ep", "--num_epoch", default=200, type=int, help="number of epochs")
    parser.add_argument("-bs", "--batch_size", default=128, type=int, help="batch size")
    parser.add_argument("-run", "--num_runs", default=1, type=int, help="number of runs")
    parser.add_argument("-gpu", "--gpu_id", default='0', type=str, help="GPU_ID: ['0', '1']")

    args = parser.parse_args()

    dataset = args.dataset # "cifar10"
    noise_rate = args.noise_rate # 0.6
    noise_type = args.noise_type # "sym"
    loss_name = args.loss_name # "cce" # "mae" # "dmi" # "cce" # "rll"
    arch = args.architecture # "inception"
    batch_norm = bool(args.batch_norm)
    weight_decay = args.weight_decay
    data_aug = bool(args.data_aug)
    num_epoch = args.num_epoch
    batch_size = args.batch_size # 128
    num_runs = args.num_runs # 1
    gpu_id = args.gpu_id

    print(f"batch_norm: {batch_norm}, weight_decay: {weight_decay}\n")

    device = torch.device('cuda:'+gpu_id if torch.cuda.is_available() else "cpu")

    mode = args.mode # "coteaching_plus" # "coteaching"

    if mode == "coteaching_plus":
        if dataset == "mnist":
            init_epoch = 5
        elif dataset == "cifar10":
            init_epoch = 10 # 20

    for run in range(num_runs):

        t_start = time.time()

        epoch_loss_train = []
        epoch_acc_train = []
        epoch_loss_test = []
        epoch_acc_test = []

        print("\n==================\n")
        print(f"== RUN No.: {run} ==")
        print("\n==================\n")

        t_start = time.time()

        chkpt_path = f"./checkpoint/{mode}/{arch}/{dataset}/{noise_type}/0{str(int(noise_rate*10))}/run_{str(run)}/"

        res_path = f"./results_pkl/{mode}/{arch}/{dataset}/{noise_type}/0{str(int(noise_rate*10))}/run_{str(run)}/"

        plt_path = f"./plots/{mode}/{arch}/{dataset}/{noise_type}/0{str(int(noise_rate*10))}/run_{str(run)}/"

        log_dirs_path = f"./runs/{mode}/{arch}/{dataset}/{noise_type}/0{str(int(noise_rate*10))}/run_{str(run)}/"

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
        print("file name: %s-bn-%s-wd-%s-aug-%s-%s-%s-%s-%s-nr-0%s-run-%s-m1.pt" % (
                    mode, batch_norm, str(weight_decay), data_aug, arch, dataset, 
                    loss_name, noise_type, str(int(noise_rate * 10)), str(run)))
        print("\n=============================\n")


        """
        Read DATA
        """

        dat, ids = read_data(noise_type, noise_rate, dataset, data_aug=data_aug, mode=mode)

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


        print("\n=============================\n")
        print("X_train: ", X_train.shape, " y_train: ", y_train.shape, "\n")
        print("X_val: ", X_val.shape, " y_val: ", y_val.shape, "\n")
        print("X_test: ", X_test.shape, " y_test: ", y_test.shape, "\n")
        print("y_train - min : {}, y_val - min : {}, \
              y_test - min : {}".format(np.min(y_train),
                                        np.min(y_val),
                                        np.min(y_test)))
        print("y_train - max : {}, y_val - max : {}, \
               y_test - max : {}".format(np.max(y_train),
                                         np.max(y_val),
                                         np.max(y_test)))
        print("\n=============================\n")
        print("\n Noise Type: {}, Noise Rate: {} \n".format(noise_type, noise_rate))

        tensor_x_train = torch.Tensor(X_train) 
        tensor_y_train = torch.Tensor(y_train)
        tensor_id_train = torch.from_numpy(np.asarray(list(range(X_train.shape[0]))))

        dataset_train = torch.utils.data.TensorDataset(tensor_x_train,
                                                       tensor_y_train,
                                                       tensor_id_train)
        train_loader = torch.utils.data.DataLoader(dataset_train,
                                                   batch_size=batch_size,
                                                   shuffle=True)

        # Val. set
        tensor_x_val = torch.Tensor(X_val)
        tensor_y_val = torch.Tensor(y_val)
        # tensor_id_val = torch.Tensor(idx_val)
        
        val_size = 1000
        dataset_val = torch.utils.data.TensorDataset(tensor_x_val,
                                                     tensor_y_val) #, tensor_id_val)
        val_loader = torch.utils.data.DataLoader(dataset_val,
                                                 batch_size=val_size,
                                                 shuffle=True)

        # Test set
        tensor_x_test = torch.Tensor(X_test)
        tensor_y_test = torch.Tensor(y_test)
        test_size = 1000
        dataset_test = torch.utils.data.TensorDataset(tensor_x_test, tensor_y_test)
        test_loader = torch.utils.data.DataLoader(dataset_test,
                                                  batch_size=test_size,
                                                  shuffle=True)



        """
        Choose MODEL and LOSS FUNCTION
        """

        if dataset == "cifar10":
            model_1 = CNN()
            model_2 = CNN()

            learning_rate = 2e-4 # 1e-3
            args.epoch_decay_start = 80
            bs_step = 313

        elif dataset == "mnist":
            model_1 = MLPNet()
            model_2 = MLPNet()

            learning_rate = 2e-4
            args.epoch_decay_start = 80
            bs_step = 375

        model_1 = model_1.to(device)
        model_2 = model_2.to(device)
        print(model_1)

        # Adjust learning rate and betas for Adam Optimizer
        mom1 = 0.9
        mom2 = 0.1
        alpha_plan = [learning_rate] * num_epoch
        beta1_plan = [mom1] * num_epoch
        for i in range(args.epoch_decay_start, num_epoch):
            alpha_plan[i] = float(num_epoch - i) / (num_epoch -
                                                    args.epoch_decay_start
                                                    ) * learning_rate
            beta1_plan[i] = mom2
        
        def adjust_learning_rate(optimizer, epoch):
            for param_group in optimizer.param_groups:
                param_group['lr']=alpha_plan[epoch]
                param_group['betas']=(beta1_plan[epoch], 0.999)
    
        kwargs = {}

        if loss_name == "rll":
            kwargs['alpha'] = 0.1 # 0.45 # 0.01
        elif loss_name == "gce":
            kwargs['q'] = 0.7
        elif loss_name == "norm_mse":
            kwargs['alpha'] = 0.1
            kwargs['beta'] = 1.

        loss_fn = get_loss(loss_name, num_class, reduction="none", **kwargs)
        
        print("\n===========\nloss: {}\n===========\n".format(loss_name))

        if loss_name == "dmi":
            model_1.load_state_dict(torch.load(chkpt_path +
                                               "%s-bn-%s-wd-%s-aug-%s-%s-%s-cce-%s-\
                                                nr-0%s-run-%s-m1.pt" % (mode, batch_norm,
                                                                        str(weight_decay),
                                                                        data_aug, arch,
                                                                        dataset, noise_type,
                                                                        str(int(noise_rate * 10)),
                                                                        str(run)))['model_state_dict'])
            model_2.load_state_dict(torch.load(chkpt_path +
                                               "%s-bn-%s-wd-%s-aug-%s-%s-%s-cce-%s-\
                                                nr-0%s-run-%s-m2.pt" % (mode, batch_norm,
                                                                        str(weight_decay),
                                                                        data_aug, arch,
                                                                        dataset, noise_type,
                                                                        str(int(noise_rate * 10)),
                                                                        str(run)))['model_state_dict'])

       
        optimizer_1 = torch.optim.Adam(model_1.parameters(), lr=learning_rate)
        optimizer_2 = torch.optim.Adam(model_2.parameters(), lr=learning_rate)

        if dataset == "cifar10":
            lr_scheduler_1 = lr_scheduler.ReduceLROnPlateau(optimizer_1, mode='min',
                        factor=0.1, patience=5, verbose=True, threshold=0.0001,
                        threshold_mode='rel', cooldown=0, min_lr=1e-5, eps=EPS)
            lr_scheduler_2 = lr_scheduler.ReduceLROnPlateau(optimizer_2, mode='min',
            factor=0.1, patience=5, verbose=True, threshold=0.0001,
            threshold_mode='rel', cooldown=0, min_lr=1e-5, eps=EPS)
            
        elif dataset == "mnist":
            lr_scheduler_1 = lr_scheduler.ReduceLROnPlateau(optimizer_1, mode='min',
                        factor=0.1, patience=5, verbose=True, threshold=0.0001,
                        threshold_mode='rel', cooldown=0, min_lr=1e-5, eps=EPS)
            lr_scheduler_2 = lr_scheduler.ReduceLROnPlateau(optimizer_2, mode='min',
            factor=0.1, patience=5, verbose=True, threshold=0.0001,
            threshold_mode='rel', cooldown=0, min_lr=1e-5, eps=EPS)



        """
        Setting up Tensorbard
        """
        writer = SummaryWriter(log_dirs_path)
        # writer.add_graph(model_1, (torch.transpose(tensor_x_train[0].unsqueeze(1),
        #                                            0,1)).type(torch.LongTensor).to(device))
        # writer.close()

        best_acc_val_1 = 0.
        best_acc_val_2 = 0.

        epoch_loss_train_1 = []
        epoch_acc_train_1 = []
        epoch_loss_test_1 = []
        epoch_acc_test_1 = []

        epoch_loss_train_2 = []
        epoch_acc_train_2 = []
        epoch_loss_test_2 = []
        epoch_acc_test_2 = []

        epoch_lab_prec_1 = []
        epoch_lab_prec_2 = []

        """
        Aggregate sample-wise values for each batch
        """
        epoch_loss_train_agg_1 = np.zeros((X_train.shape[0], num_epoch))
        epoch_acc_train_agg_1 = np.zeros((X_train.shape[0], num_epoch))
        epoch_pred_train_agg_1 = np.zeros((X_train.shape[0], num_epoch))
        epoch_idx_sel_tr_agg_1 = np.zeros((X_train.shape[0], num_epoch))

        epoch_loss_train_agg_2 = np.zeros((X_train.shape[0], num_epoch))
        epoch_acc_train_agg_2 = np.zeros((X_train.shape[0], num_epoch))
        epoch_pred_train_agg_2 = np.zeros((X_train.shape[0], num_epoch))
        epoch_idx_sel_tr_agg_2 = np.zeros((X_train.shape[0], num_epoch))


        noise_or_not = y_train == y_temp[idx_train]

        rate_ep = rate_schedule(args, num_gradual=10, exponent=1)

        for epoch in range(1, num_epoch+1):

            (loss_train_1, acc_train_1, loss_train_agg_1, acc_train_agg_1,
            pred_train_agg_1, loss_train_2, acc_train_2, loss_train_agg_2,
            acc_train_agg_2, pred_train_agg_2, lab_prec_1, lab_prec_2,
            idx_sel_tr_agg_1, idx_sel_tr_agg_2) = coteaching_train(epoch, train_loader,
                                                                   model_1, model_2,
                                                                   rate_ep[epoch-1],
                                                                   noise_or_not,
                                                                   loss_fn)

            # Logs
            writer.add_scalar('training_loss_1', loss_train_1, epoch)
            writer.add_scalar('training_accuracy_1', acc_train_1, epoch)
            writer.add_scalar('training_loss_2', loss_train_2, epoch)
            writer.add_scalar('training_accuracy_2', acc_train_2, epoch)
            writer.close()          

            # Validation
            loss_val_1, acc_val_1 = test(val_loader, model_1, run, use_best=False)
            loss_val_2, acc_val_2 = test(val_loader, model_2, run, use_best=False)

            # Testing
            loss_test_1, acc_test_1 = test(test_loader, model_1, run, use_best=False)
            loss_test_2, acc_test_2 = test(test_loader, model_2, run, use_best=False)

            # Logs
            writer.add_scalar('testing_loss_1', loss_test_1, epoch)
            writer.add_scalar('testing_accuracy_1', acc_test_1, epoch)
            writer.add_scalar('testing_loss_2', loss_test_2, epoch)
            writer.add_scalar('testing_accuracy_2', acc_test_2, epoch)
            writer.close()

            if dataset == "mnist":
                lr_scheduler_1.step(loss_val_1)
                lr_scheduler_2.step(loss_val_2)
            elif dataset == "cifar10":
                lr_scheduler_1.step(loss_val_1)
                lr_scheduler_2.step(loss_val_2)
        

            epoch_loss_train_1.append(loss_train_1)
            epoch_acc_train_1.append(acc_train_1)
            epoch_idx_sel_tr_agg_1[:, epoch - 1] = idx_sel_tr_agg_1

            epoch_loss_train_2.append(loss_train_2)
            epoch_acc_train_2.append(acc_train_2)
            epoch_idx_sel_tr_agg_2[:, epoch - 1] = idx_sel_tr_agg_2

            epoch_loss_test_1.append(loss_test_1)
            epoch_acc_test_1.append(acc_test_1)
            epoch_loss_test_2.append(loss_test_2)
            epoch_acc_test_2.append(acc_test_2)
            epoch_lab_prec_1.append(lab_prec_1)
            epoch_lab_prec_2.append(lab_prec_2)

            if epoch == 0:
                best_acc_val_1 = acc_val_1
                best_acc_val_2 = acc_val_2
            
            if best_acc_val_1 < acc_val_1:

                best_model_wts = copy.deepcopy(model_1.state_dict())
                state_dict = {'model_state_dict': model_1.state_dict(), 
                              'opt_state_dict': optimizer_1.state_dict(),
                              'best_acc_val': best_acc_val_1,
                              'epoch': epoch, 'run': run}
                torch.save(state_dict, chkpt_path + "%s-bn-%s-wd-%s-aug-%s-%s-%s-%s-%s-nr-0%s-run-%s-m1.pt" % (
                                        mode, batch_norm, str(weight_decay), data_aug, arch, dataset, 
                                        loss_name, noise_type, str(int(noise_rate * 10)), str(run)))
                print("Best model weights updated...\n")
                best_acc_val_1 = acc_val_1

            if best_acc_val_2 < acc_val_2:

                best_model_wts = copy.deepcopy(model_2.state_dict())
                state_dict = {'model_state_dict': model_2.state_dict(), 
                              'opt_state_dict': optimizer_2.state_dict(),
                              'best_acc_val': best_acc_val_2,
                              'epoch': epoch, 'run': run}
                torch.save(state_dict, chkpt_path + "%s-bn-%s-wd-%s-aug-%s-%s-%s-%s-%s-nr-0%s-run-%s-m2.pt" % (
                                        mode, batch_norm, str(weight_decay), data_aug, arch, dataset, 
                                        loss_name, noise_type, str(int(noise_rate * 10)), str(run)))
                print("Best model weights updated...\n")
                best_acc_val_2 = acc_val_2

            print("::: Model - 1 :::\n")
            print("Epoch: {}, lr: {}, loss_train: {}, loss_val: {}, loss_test: {:.3f},\
                   acc_train: {}, acc_val: {}, acc_test: {:.3f}\n".format(epoch,
                                                                          optimizer_1.param_groups[0]['lr'],
                                                                          loss_train_1, loss_val_1,
                                                                          loss_test_1, acc_train_1,
                                                                          acc_val_1, acc_test_1))
            print("::: Model - 2 :::\n")
            print("Epoch: {}, lr: {}, loss_train: {}, loss_val: {}, loss_test: {:.3f},\
                   acc_train: {}, acc_val: {}, acc_test: {:.3f}\n".format(epoch,
                                                                          optimizer_2.param_groups[0]['lr'],
                                                                          loss_train_2, loss_val_2,
                                                                          loss_test_2, acc_train_2,
                                                                          acc_val_2, acc_test_2))


        loss_test_1, acc_test_1 = test(test_loader, model_1, run, use_best=False)
        loss_test_2, acc_test_2 = test(test_loader, model_2, run, use_best=False)

        print(f"Model - 1::: Run: {run}::: Test set performance - \
              test_acc: {acc_test_1}, test_loss: {loss_test_1}\n")
        print(f"Model - 2::: Run: {run}::: Test set performance - \
              test_acc: {acc_test_2}, test_loss: {loss_test_2}\n")

        if noise_rate > 0.:
            state_dict = {'model_state_dict': model_1.state_dict(),
                          'opt_state_dict': optimizer_1.state_dict(),
                          'best_acc_val': best_acc_val_1,
                          'epoch': epoch,
                          'run': run}
            torch.save(state_dict, chkpt_path + 
                        "%s-bn-%s-wd-%s-aug-%s-%s-%s-%s-%s-nr-0%s-run-%s-m1.pt" % (mode,
                                                                                   batch_norm,
                                                                                   str(weight_decay),
                                                                                   data_aug, arch,
                                                                                   dataset,
                                                                                   loss_name,
                                                                                   noise_type,
                                                                                   str(int(noise_rate * 10)),
                                                                                   str(run)))

            state_dict = {'model_state_dict': model_2.state_dict(),
                          'opt_state_dict': optimizer_2.state_dict(),
                          'best_acc_val': best_acc_val_2,
                          'epoch': epoch, 'run': run}
            torch.save(state_dict, chkpt_path + 
                       "%s-bn-%s-wd-%s-aug-%s-%s-%s-%s-%s-nr-0%s-run-%s-m2.pt" % (mode,
                                                                                  batch_norm,
                                                                                  str(weight_decay),
                                                                                  data_aug, arch,
                                                                                  dataset, loss_name,
                                                                                  noise_type,
                                                                                  str(int(noise_rate * 10)),
                                                                                  str(run)))
        # Print the elapsed time
        elapsed = time.time() - t_start
        print("\nelapsed time: \n", elapsed)

        """
        Save results
        """

        with open(res_path + 
                  "%s-bn-%s-wd-%s-aug-%s-%s-%s-%s-%s-nr-0%s-run-%s.pickle" % (mode, batch_norm,
                                                                              str(weight_decay),
                                                                              data_aug, arch,
                                                                              dataset, loss_name,
                                                                              noise_type,
                                                                              str(int(noise_rate * 10)),
                                                                              str(run)), 'wb') as f:
            if noise_rate > 0.:
                pickle.dump({'epoch_loss_train_1': np.asarray(epoch_loss_train_1),
                             'epoch_acc_train_1': np.asarray(epoch_acc_train_1),
                             'epoch_loss_test_1': np.asarray(epoch_loss_test_1),
                             'epoch_acc_test_1': np.asarray(epoch_acc_test_1),
                             'epoch_loss_train_2': np.asarray(epoch_loss_train_2),
                             'epoch_acc_train_2': np.asarray(epoch_acc_train_2),
                             'epoch_loss_test_2': np.asarray(epoch_loss_test_2),
                             'epoch_acc_test_2': np.asarray(epoch_acc_test_2),
                             'idx_tr_clean_ref': idx_tr_clean_ref,
                             'idx_tr_noisy_ref': idx_tr_noisy_ref,
                             'y_train_org': y_temp[idx_train],
                             'y_train':y_train,
                             'epoch_lab_prec_1': epoch_lab_prec_1,
                             'epoch_lab_prec_2': epoch_lab_prec_2,
                             'epoch_idx_sel_tr_agg_1':epoch_idx_sel_tr_agg_1,
                             'epoch_idx_sel_tr_agg_2':epoch_idx_sel_tr_agg_2,
                             'num_epoch': num_epoch,
                             'time_elapsed': elapsed}, f,
                            protocol=pickle.HIGHEST_PROTOCOL)
            else:
                pickle.dump({'epoch_loss_train_1': np.asarray(epoch_loss_train_1),
                             'epoch_acc_train_1': np.asarray(epoch_acc_train_1),
                             'epoch_loss_test_1': np.asarray(epoch_loss_test_1),
                             'epoch_acc_test_1': np.asarray(epoch_acc_test_1),
                             'epoch_loss_train_2': np.asarray(epoch_loss_train_2),
                             'epoch_acc_train_2': np.asarray(epoch_acc_train_2),
                             'epoch_loss_test_2': np.asarray(epoch_loss_test_2),
                             'epoch_acc_test_2': np.asarray(epoch_acc_test_2),
                             'y_train':y_train,
                             'epoch_idx_sel_tr_agg_1':epoch_idx_sel_tr_agg_1,
                             'epoch_idx_sel_tr_agg_2':epoch_idx_sel_tr_agg_2,
                             'num_epoch': num_epoch, 'time_elapsed': elapsed}, f,
                            protocol=pickle.HIGHEST_PROTOCOL)
        
        print("Pickle file saved: " + res_path +
              "%s-bn-%s-wd-%s-aug-%s-%s-%s-%s-%s-nr-0%s-run-%s.pickle" % (mode, batch_norm,
                                                                          str(weight_decay),
                                                                          data_aug, arch,
                                                                          dataset, loss_name,
                                                                          noise_type,
                                                                          str(int(noise_rate * 10)),
                                                                          str(run)), "\n")
