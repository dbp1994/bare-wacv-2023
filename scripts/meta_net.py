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
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter

from data import read_data
from losses import get_loss
from meta_layers import MLPNet, Meta_CNN, MetaModule, MetaLinear

# set seed for reproducibility
torch.manual_seed(1337)
np.random.seed(3459)
# torch.cuda.manual_seed_all(3459)
# tf.set_random_seed(3459)

torch.autograd.set_detect_anomaly(True)

eps = 1e-8

def accuracy(true_label, pred_label):
	num_samples = true_label.shape[0]
	err = [1 if (pred_label[i] != true_label[i]).sum() == 0 else 0 for i in range(num_samples)]
	acc = 1 - (sum(err)/num_samples)
	return acc


# Meta-weight Net
class MetaWeightNet(MetaModule):
    def __init__(self, input, hidden1, output):
        super(MetaWeightNet, self).__init__()
        self.linear1 = MetaLinear(input, hidden1)
        self.linear11 = MetaLinear(hidden1, hidden1)
        self.linear2 = MetaLinear(hidden1, output)
        # self.linear3 = MetaLinear(hidden2, output)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear11(x)
        x = F.relu(x)
        out = self.linear2(x)
        return torch.sigmoid(out)

def meta_net_train(train_loader, X_val, y_val, model, meta_wt_net):

    loss_train = 0.
    acc_train = 0.
    correct = 0

    sample_wts = np.zeros((len(train_loader.dataset), ))

    model.train()
    meta_wt_net.train()

    loss_train_agg = np.zeros((len(train_loader.dataset), ))
    acc_train_agg = np.zeros((len(train_loader.dataset), ))
    pred_train_agg = np.zeros((len(train_loader.dataset), ))

    for batch_id, (x, y, idx) in tqdm(enumerate(train_loader)):

        y = y.type(torch.LongTensor)
        y_val = y_val.type(torch.LongTensor)

        if dataset == "news":
            x = x.type(torch.LongTensor)
        
        # Transfer data to the GPU
        x, y, idx = x.to(device), y.to(device), idx.to(device)
        x_val, y_val = X_val.to(device), y_val.to(device)

        if dataset == "mnist":
            meta_net = MLPNet()
        elif dataset == "cifar10":
            meta_net = Meta_CNN()
        meta_net = meta_net.to(device)
        meta_net.load_state_dict(model.state_dict())

        y_f_hat = meta_net(x)
        loss_int = loss_fn(y_f_hat, y)
        loss_int = loss_int.reshape((len(loss_int), 1))
        
        ex_wts_tmp = meta_wt_net(loss_int)
        loss_tmp = torch.mean(loss_int * ex_wts_tmp)
        meta_net.zero_grad()
        grads = torch.autograd.grad(loss_tmp, meta_net.params(), create_graph=True)

        # For ResNets
        ## interim_lr = learning_rate * (0.1**int(epoch >= 80)) * (0.1**int(epoch >= 100))

        meta_net.update_params(lr_inner=meta_lr, source_params=grads)
        y_wts_hat = meta_net(x_val)
        loss_wts = loss_fn(y_wts_hat, y_val)
        optimizer_wts.zero_grad()
        loss_wts.mean().backward()
        optimizer_wts.step()


        output = model(x)
        loss_batch = loss_fn(output, y)
        loss_batch = loss_batch.reshape((len(loss_batch), 1))

        with torch.no_grad():
            ex_wts = meta_wt_net(loss_batch)

        norm_w = torch.sum(ex_wts)
        if norm_w >= 1e-8:
            ex_wts /= norm_w
            
        loss_meta_net = torch.mean(ex_wts * loss_batch)
        optimizer.zero_grad()
        loss_meta_net.backward()
        optimizer.step()
        
        # Compute the accuracy and loss for model_stud
        pred_prob = F.softmax(output, dim=1)
        pred = torch.argmax(pred_prob, dim=1)
        loss_train += loss_meta_net.item()
        correct += pred.eq(y.to(device)).sum().item()

        batch_cnt = batch_id + 1

        loss_train_agg[list(map(int, idx.tolist()))] = np.asarray(loss_meta_net.tolist())
        acc_train_agg[list(map(int, idx.tolist()))] = np.asarray(pred.eq(y.to(device)).tolist())
        pred_train_agg[list(map(int, idx.tolist()))] = np.asarray(pred.tolist())

        sample_wts[list(map(int, (idx.to('cpu')).tolist()))] = np.asarray((ex_wts.to('cpu').reshape(ex_wts.shape[0],)).tolist())

    loss_train /= batch_cnt
    acc_train = 100.*correct/len(train_loader.dataset)

    return sample_wts, loss_train, acc_train, loss_train_agg, acc_train_agg, pred_train_agg

def test(data_loader, model, run, use_best=False):

    loss_test = 0.
    correct = 0

    model.eval()

    with torch.no_grad():
        for batch_id, (x, y) in enumerate(data_loader):
            if use_best == True:
                # load best model weights
                model.load_state_dict(torch.load("%s-%s-%s-%s-nr-0%s-mdl-wts-\
                                                 run-%s.pt" % (mode, dataset, loss_name,
                                                               noise_type, str(int(noise_rate
                                                               * 10)), str(run))))
                model = model.to(device)

            y = y.type(torch.LongTensor)
            x, y = x.to(device), y.to(device)

            output = model(x)
            pred_prob = F.softmax(output, dim=1)
            pred = torch.argmax(pred_prob, dim=1)

            loss_test += torch.mean(loss_fn(output, y)).item()
            correct += (pred.eq(y.to(device))).sum().item()

            batch_cnt = batch_id + 1
        
    loss_test /= batch_cnt
    acc_test = 100.*correct/len(data_loader.dataset)

    return loss_test, acc_test


"""
Configuration
"""

parser = argparse.ArgumentParser(description='Meta Weight Net - NeurIPS 2019')
parser.add_argument('-dat', '--dataset', default="mnist",type=str, help="dataset")
parser.add_argument('-nr', '--noise_rate', default=0.4, type=float, help="noise rate")
parser.add_argument('-nt', '--noise_type', default="sym", type=str, help="noise type")
parser.add_argument('-loss', '--loss_name', default="cce",type=str, help="loss name")
parser.add_argument('-da', '--data_aug', default=0, type=int, help="data augmentation (0 or 1)")
parser.add_argument("-bs", "--batch_size", default=128, type=int, help="batch size")
parser.add_argument('-ep', '--num_epoch', default=100, type=int, help="number of epochs")
parser.add_argument('-run', '--num_runs', default=1, type=int, help="number of runs/simulations")
args = parser.parse_args()


dataset = args.dataset
noise_rate = args.noise_rate
noise_type = args.noise_type
data_aug = bool(args.data_aug)
batch_size = args.batch_size
loss_name = args.loss_name
num_epoch = args.num_epoch
num_runs = args.num_runs

random_state = 422
learning_rate = 2e-4
mode = "meta_net"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

"""
Meta_Net hyper-params
"""
meta_lr = 1e-4 #1e-3 


for run in range(num_runs):

    t_start = time.time()

    print("\n==================\n")
    print(f"== RUN No.: {run} ==")
    print("\n==================\n")

    chkpt_path = "./checkpoint/" + mode + "/" + dataset + "/" + noise_type + \
                 "/0" + str(int(noise_rate*10)) + "/run_" + str(run) + "/"

    res_path = "./results_pkl/" + mode + "/" + dataset + "/" + noise_type + \
               "/0" + str(int(noise_rate*10)) + "/run_" + str(run) + "/"

    plt_path = "./plots/" + mode + "/" + dataset + "/" + noise_type + \
               "/0" + str(int(noise_rate*10)) + "/run_" + str(run) + "/"

    log_dirs_path = "./runs/" + mode + "/" + dataset + "/" + noise_type + \
                    "/0" + str(int(noise_rate*10)) + "/run_" + str(run) + "/"

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
    print("file name: %s-%s-cce-%s-nr-0%s-mdl-wts-run-%s.pt" % (mode, dataset,
                                                                noise_type,
                                                                str(int(noise_rate
                                                                * 10)), str(run)))
    print("\n=============================\n")

    """
    Read DATA
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
    print("y_train - min : {}, y_val - min : {}, y_test - \
          min : {}".format(np.min(y_train), np.min(y_val), np.min(y_test)))
    print("y_train - max : {}, y_val - max : {}, y_test - \
          max : {}".format(np.max(y_train), np.max(y_val), np.max(y_test)))
    print("\n=============================\n")
    print("\n Noise Type: {}, Noise Rate: {} \n".format(noise_type, noise_rate))

    """
    Create Dataset Loader
    """

    tensor_x_train = torch.Tensor(X_train) # .as_tensor() avoids copying, .Tensor() creates a new copy
    tensor_y_train = torch.Tensor(y_train) # .as_tensor() avoids copying, .Tensor() creates a new copy
    tensor_id_train = torch.Tensor(np.asarray(list(range(X_train.shape[0]))))

    dataset_train = torch.utils.data.TensorDataset(tensor_x_train, tensor_y_train,
                                                   tensor_id_train)
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size,
                                               shuffle=True)

    tensor_x_val = torch.Tensor(X_val)
    tensor_y_val = torch.Tensor(y_val)

    val_size = 100
    dataset_val = torch.utils.data.TensorDataset(tensor_x_val, tensor_y_val)
    val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=val_size,
                                             shuffle=True)

    tensor_x_test = torch.Tensor(X_test)
    tensor_y_test = torch.Tensor(y_test)

    test_size = 100
    dataset_test = torch.utils.data.TensorDataset(tensor_x_test, tensor_y_test)
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=test_size,
                                              shuffle=True)

    """
    Choose MODEL and LOSS FUNCTION
    """
    
    if dataset == "mnist":
        model = MLPNet()
        meta_wt_net = MetaWeightNet(1, 50, 1)
    elif dataset in ["cifar10", "cifar100"]:
        model = Meta_CNN()
        meta_wt_net = MetaWeightNet(1, 50, 1)

    model = model.to(device)
    meta_wt_net = meta_wt_net.to(device)
    print(model)
    print(meta_wt_net)

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

    if loss_name == "dmi":
        model.load_state_dict(torch.load(chkpt_path + "%s-%s-cce-%s-nr-0%s-mdl-wts-\
                                         run-%s.pt"% (mode, dataset, noise_type,
                                         str(int(noise_rate * 10)), str(run))))

    optimizer = optim.Adam(model.params(), lr=learning_rate)
    if dataset == "mnist":
        optimizer_wts = optim.SGD(meta_wt_net.params(), lr = 1e-3, momentum=0.9)
    elif dataset in ["cifar10", "cifar100"]:
        optimizer_wts = optim.Adam(meta_wt_net.params(), lr = 1e-4)

    lr_scheduler_1 = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                factor=0.1, patience=5, verbose=True, threshold=0.0001,
                threshold_mode='rel', cooldown=0, min_lr=1e-5, eps=1e-08)
    lr_scheduler_2 = lr_scheduler.MultiStepLR(optimizer, milestones=[30,80], gamma=0.1)
    # lr_scheduler_2 = lr_scheduler.MultiStepLR(optimizer, milestones=[100,150], gamma=0.1)
    lr_scheduler_3 = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    """
    Setting up Tensorbard
    """
    writer = SummaryWriter(log_dirs_path)
    writer.add_graph(model, (torch.transpose(tensor_x_train[0].unsqueeze(1),0,1)).to(device))
    writer.close()

    best_acc_val = 0.

    epoch_loss_train = []
    epoch_acc_train = []
    epoch_loss_test = []
    epoch_acc_test = []

    epoch_loss_train_agg = np.zeros((len(train_loader.dataset), num_epoch))
    epoch_acc_train_agg = np.zeros((len(train_loader.dataset), num_epoch))
    epoch_pred_train_agg = np.zeros((len(train_loader.dataset), num_epoch))

    sample_wts_fin = np.zeros((len(train_loader.dataset), num_epoch))

    t_start = time.time()

    for epoch in range(num_epoch):

        #Training set performance
        sample_wts, loss_train, acc_train, loss_train_agg, \
        acc_train_agg, pred_train_agg = meta_net_train(train_loader, tensor_x_val,
                                                       tensor_y_val, model,
                                                       meta_wt_net)
        writer.add_scalar('training_loss', loss_train, epoch)
        writer.add_scalar('training_accuracy', acc_train, epoch)
        writer.close()
        

        # Validation set performance
        loss_val, acc_val = test(val_loader, model, run, use_best=False)

        #Test set performance
        loss_test, acc_test = test(test_loader, model, run, use_best=False)
        writer.add_scalar('test_loss', loss_test, epoch)
        writer.add_scalar('test_accuracy', acc_test, epoch)
        writer.close()

        epoch_loss_train.append(loss_train)
        epoch_acc_train.append(acc_train)
        epoch_loss_train_agg[:, epoch] = loss_train_agg
        epoch_acc_train_agg[:, epoch] = acc_train_agg
        epoch_pred_train_agg[:, epoch] = pred_train_agg
        sample_wts_fin[:, epoch] = sample_wts

        epoch_loss_test.append(loss_test)
        epoch_acc_test.append(acc_test)

        # Learning Rate Scheduler Update
        lr_scheduler_1.step(loss_val)
        ##lr_scheduler_3.step()
        # lr_scheduler_2.step()

        # Update best_acc_val and sample_wts_fin
        if epoch == 0:
            best_acc_val = acc_val

        if acc_val > best_acc_val:
            best_acc_val = acc_val
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), chkpt_path +
                       "%s-%s-%s-%s-nr-0%s-mdl-wts-run-%s.pt" % (mode, dataset, loss_name,
                                                                 noise_type, str(int(noise_rate
                                                                 * 10)), str(run)))
            print("Model weights updated...\n")

        print("Epoch: {}, lr: {}, loss_train: {}, loss_val: {}, loss_test: {:.3f}, \
              acc_train: {}, acc_val: {}, acc_test: {:.3f}\n".format(epoch,
                                                                     optimizer.param_groups[0]['lr'],
                                                                     loss_train, loss_val,
                                                                     loss_test,acc_train,
                                                                     acc_val, acc_test))


    loss_test, acc_test = test(test_loader, model, run, use_best=False)

    print(f"Run: {run}::: Test set performance - test_acc: {acc_test}, test_loss: {loss_test}\n")

    if noise_rate > 0.:
        torch.save(model.state_dict(), chkpt_path + 
                   "%s-%s-%s-%s-nr-0%s-mdl-wts-run-%s.pt" % (mode, dataset, loss_name,
                                                             noise_type, str(int(noise_rate
                                                             * 10)), str(run)))


    # Print the elapsed time
    elapsed = time.time() - t_start
    print("\nelapsed time: \n", elapsed)

    """
    Save results
    """
    with open(res_path+ "%s-%s-%s-%s-nr-0%s-\
              run-%s.pickle" % (mode, dataset, loss_name, noise_type,
                                str(int(noise_rate * 10)), str(run)), 'wb') as f:

        if noise_rate > 0:
            pickle.dump({'sample_wts_fin': sample_wts_fin, 
                        'epoch_loss_train': np.asarray(epoch_loss_train), 
                        'epoch_acc_train': np.asarray(epoch_acc_train), 
                        'epoch_loss_test': np.asarray(epoch_loss_test), 
                        'epoch_acc_test': np.asarray(epoch_acc_test),
                        'idx_tr_clean_ref': idx_tr_clean_ref, 
                        'idx_tr_noisy_ref': idx_tr_noisy_ref, 
                        'y_train_org': y_temp[idx_train], 'y_train':y_train, 
                        'num_epoch': num_epoch,
                        'time_elapsed': elapsed}, f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            pickle.dump({'sample_wts_fin': sample_wts_fin, 
                        'epoch_loss_train': np.asarray(epoch_loss_train), 
                        'epoch_acc_train': np.asarray(epoch_acc_train), 
                        'epoch_loss_test': np.asarray(epoch_loss_test), 
                        'epoch_acc_test': np.asarray(epoch_acc_test),
                        'num_epoch': num_epoch,
                        'time_elapsed': elapsed}, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("Pickle file saved: " + res_path + \
          "%s-%s-%s-%s-nr-0%s-run-%s.pickle" % (mode, dataset, loss_name,
                                                noise_type, str(int(noise_rate
                                                * 10)), str(run)), "\n")