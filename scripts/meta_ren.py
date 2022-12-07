from __future__ import print_function, absolute_import

import os
import time
import pathlib
import pickle
from tqdm import tqdm
import copy
import argparse

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter

from data import read_data
from meta_layers import MNIST_MetaNN, MLPNet, Meta_CNN
from losses import get_loss

# set seed for reproducibility
torch.manual_seed(1337)
np.random.seed(3459)
# torch.cuda.manual_seed_all(3459)
# tf.set_random_seed(3459)

eps = 1e-8


def accuracy(true_label, pred_label):
	num_samples = true_label.shape[0]
	err = [1 if (pred_label[i] != true_label[i]).sum()==0 else 0 for i in range(num_samples)]
	acc = 1 - (sum(err)/num_samples)
	return acc


def meta_ren_train(train_loader, X_val, y_val, model):

    loss_train = 0.
    acc_train = 0.
    correct = 0

    # meta_lr = 3e-4
    sample_wts = np.zeros((len(train_loader.dataset), ))

    loss_train_agg = np.zeros((len(train_loader.dataset), ))
    acc_train_agg = np.zeros((len(train_loader.dataset), ))
    pred_train_agg = np.zeros((len(train_loader.dataset), ))

    model.train()

    for batch_id, (x, y, idx) in tqdm(enumerate(train_loader)):

        y = y.type(torch.LongTensor)
        y_val = y_val.type(torch.LongTensor)
        
        # Transfer data to the GPU
        x, y, idx = x.to(device), y.to(device), idx.to(device)
        x_val, y_val = X_val.to(device), y_val.to(device)

        # Load the current n/w params. into meta_net
        if dataset == "mnist":
            meta_net = MLPNet()
        elif dataset == "cifar10":
            meta_net = Meta_CNN() 

        meta_net = meta_net.to(device)
        meta_net.load_state_dict(model.state_dict())

        # Lines 4 - 5 initial forward pass to compute the initial weighted loss
        y_f_hat = meta_net(x)
        loss_pass_1 = loss_fn(y_f_hat, y)
        eps = torch.zeros(loss_pass_1.size(), requires_grad = True)
        eps = eps.to(device)
        l_f_meta = torch.sum(loss_pass_1 * eps)

        meta_net.zero_grad()

        # Line 6 - 7 perform a parameter update
        grads = torch.autograd.grad(l_f_meta, meta_net.params(), create_graph=True)
        meta_net.update_params(meta_lr, source_params=grads)

        # Line 8 - 10 2nd forward pass and getting the gradients with respect to epsilon
        y_g_hat = meta_net(x_val)
        # l_g_meta = torch.mean(cce_loss(y_g_hat, y_val))
        l_g_meta = torch.mean(loss_fn(y_g_hat, y_val))
        grad_eps = torch.autograd.grad(l_g_meta, eps, only_inputs=True)[0]

        # Line 11 computing and normalizing the weights
        w_tilde = torch.clamp(torch.neg(grad_eps), min=0)
        norm_const = torch.sum(w_tilde) 

        if norm_const != 0:
            if norm_const < 1e-8:
                ex_wts = w_tilde / 1e-8
            else:
                ex_wts = w_tilde / norm_const
        else:
            ex_wts = w_tilde

        sample_wts[list(map(int, (idx.to('cpu')).tolist()))] = np.asarray((ex_wts.to('cpu')).tolist())

        # print("\n norm_const: {}\n".format(norm_const))
        # print("\n ex_wts: {}\n".format(ex_wts))
        # print("\n====================\n")
        # input("Press <ENTER> to continue.\n")

        # Lines 12 - 14 computing for the loss with the computed weights
        # and then perform a parameter update

        y_f_hat = model(x)
        loss_batch = loss_fn(y_f_hat, y)
        loss_meta_ren = torch.sum(loss_batch * ex_wts)

        optimizer.zero_grad()
        loss_meta_ren.backward()
        optimizer.step()

        # Compute the accuracy and loss after meta-updation
        pred_prob = F.softmax(y_f_hat, dim=1)
        pred = torch.argmax(pred_prob, dim=1)
        loss_train += (loss_meta_ren).item() # .item() for scalars, .tolist() in general
        correct += pred.eq(y.to(device)).sum().item()

        loss_train_agg[list(map(int, idx.tolist()))] = np.asarray(loss_batch.tolist())
        acc_train_agg[list(map(int, idx.tolist()))] = np.asarray(pred.eq(y.to(device)).tolist())
        pred_train_agg[list(map(int, idx.tolist()))] = np.asarray(pred.tolist())

        batch_cnt = batch_id + 1

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
                model.load_state_dict(torch.load(chkpt_path + 
                                                 "%s-%s-%s-%s-nr-0%s-mdl-wts-run-%s.pt" % (mode,
                                                                                           dataset,
                                                                                           loss_name,
                                                                                           noise_type,
                                                                                           str(int(noise_rate
                                                                                           * 10)),
                                                                                           str(run))))
                model = model.to(device)

            """
            Loss Function expects the labels to be 
            integers and not floats
            """
            y = y.type(torch.LongTensor)        
            x, y = x.to(device), y.to(device)
            output = model(x)
            pred_prob = F.softmax(output, dim=1)
            pred = torch.argmax(pred_prob, dim=1)

            batch_loss = loss_fn(output, y)
            loss_test += torch.mean(batch_loss).item()
            correct += (pred.eq(y.to(device))).sum().item()

            batch_cnt = batch_id + 1
        
    loss_test /= batch_cnt
    acc_test = 100.*correct/len(data_loader.dataset)

    return loss_test, acc_test


"""
Configuration
"""

parser = argparse.ArgumentParser(description = 'Meta Reweight- Ren et al. (ICML 18)')
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
meta_lr = 2e-3
learning_rate = 2e-4
mode = "meta_ren"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


for run in range(num_runs):

    t_start = time.time()

    print("\n==================\n")
    print(f"== RUN No.: {run} ==")
    print("\n==================\n")

    epoch_loss_train = []
    epoch_acc_train = []
    epoch_loss_test = []
    epoch_acc_test = []
    sample_wts_fin = []

    t_start = time.time()

    chkpt_path = "./checkpoint/" + mode + "/" + dataset + "/" + \
                 noise_type + "/0" + str(int(noise_rate*10)) + "/" + \
                 "run_" + str(run) + "/"

    res_path = "./results_pkl/" + mode + "/" + dataset + "/" + noise_type + \
               "/0" + str(int(noise_rate*10)) + "/" + "run_" + str(run) + "/"

    plt_path = "./plots/" + mode + "/" + dataset + "/" + noise_type + \
               "/0" + str(int(noise_rate*10)) + "/" + "run_" + str(run) + "/"

    log_dirs_path = "./runs/" + mode + "/" + dataset + "/" + noise_type + \
                    "/0" + str(int(noise_rate*10)) + "/" + "run_" + str(run) + "/"

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
    print("chkpt_path: {}".format(chkpt_path))
    print("res_path: {}".format(res_path))
    print("plt_path: {}".format(plt_path))
    print("log_dirs_path: {}".format(log_dirs_path))
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
    print("y_train - min : {}, y_val - min : {}, \
          y_test - min : {}".format(np.min(y_train),
                                    np.min(y_val), np.min(y_test)))
    print("y_train - max : {}, y_val - max : {}, \
          y_test - max : {}".format(np.max(y_train),
                                    np.max(y_val), np.max(y_test)))
    print("\n=============================\n")
    print("\n Noise Type: {}, Noise Rate: {} \n".format(noise_type, noise_rate))

    """
    Create Dataset Loader
    """

    # Train. set
    tensor_x_train = torch.Tensor(X_train) # .as_tensor() avoids copying, .Tensor() creates a new copy
    tensor_y_train = torch.Tensor(y_train) # .as_tensor() avoids copying, .Tensor() creates a new copy
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

    if dataset == "mnist":
        model = MLPNet()
    elif dataset == "cifar10":
        model = Meta_CNN()
        # learning_rate = 1e-3
        # meta_lr = 2e-4

    model = model.to(device)
    print(model)
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
        model.load_state_dict(torch.load(chkpt_path + 
                              "%s-%s-cce-%s-nr-0%s-mdl-wts-run-%s.pt" % (mode, dataset,
                                                                         noise_type,
                                                                         str(int(noise_rate
                                                                         * 10)),
                                                                         str(run))))

    optimizer = optim.Adam(model.params(), lr = learning_rate)
    lr_scheduler_1 = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                factor=0.1, patience=5, verbose=True, threshold=0.0001,
                threshold_mode='rel', cooldown=0, min_lr=1e-5, eps=1e-08)
    if dataset == "mnist":
        lr_scheduler_2 = lr_scheduler.MultiStepLR(optimizer, milestones=[30,80], gamma=0.1)
    elif dataset == "cifar10":
        lr_scheduler_2 = lr_scheduler.MultiStepLR(optimizer, milestones=[80,120], gamma=0.1)
    lr_scheduler_3 = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)


    """
    Setting up Tensorbard
    """
    writer = SummaryWriter(log_dirs_path)
    writer.add_graph(model, (torch.transpose(tensor_x_train[0].unsqueeze(1),0,1)).to(device))
    writer.close()

    """
    Aggregate sample-wise loss values for each epoch
    """
    epoch_loss_train_agg = np.zeros((len(train_loader.dataset), num_epoch))
    epoch_acc_train_agg = np.zeros((len(train_loader.dataset), num_epoch))
    epoch_pred_train_agg = np.zeros((len(train_loader.dataset), num_epoch))

    sample_wts_fin = np.zeros((len(train_loader.dataset), num_epoch))

    for epoch in range(num_epoch):

        #Training set performance
        sample_wts, loss_train, acc_train, loss_train_agg, \
        acc_train_agg, pred_train_agg = meta_ren_train(train_loader, tensor_x_val,
                                                       tensor_y_val, model)

        ## log TRAIN. SET performance
        writer.add_scalar('training_loss', loss_train, epoch)
        writer.add_scalar('training_accuracy', acc_train, epoch)
        writer.close()

        # Validation set performance
        loss_val, acc_val = test(val_loader, model, run, use_best=False)

        #Testing set performance
        loss_test, acc_test = test(test_loader, model, run, use_best=False)

        ## log TEST SET performance
        writer.add_scalar('test_loss', loss_test, epoch)
        writer.add_scalar('test_accuracy', acc_test, epoch)
        writer.close()


        # Learning Rate Scheduler Update
        if dataset in ["mnist", "svhn"]:
            lr_scheduler_1.step(loss_val)
        elif dataset in ["cifar10", "cifar100"]:
            lr_scheduler_1.step(loss_val)
            ##lr_scheduler_3.step()
            ## lr_scheduler_2.step()

        epoch_loss_train.append(loss_train)
        epoch_acc_train.append(acc_train)
        epoch_loss_train_agg[:, epoch] = loss_train_agg
        epoch_acc_train_agg[:, epoch] = acc_train_agg
        epoch_pred_train_agg[:, epoch] = pred_train_agg    
        sample_wts_fin[:, epoch] = sample_wts

        epoch_loss_test.append(loss_test)
        epoch_acc_test.append(acc_test)

        # Update best_acc_val and sample_wts_fin
        if epoch == 0:
            best_acc_val = acc_val

        if acc_val > best_acc_val:
            best_acc_val = acc_val
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), chkpt_path +
                       "%s-%s-%s-%s-nr-0%s-mdl-wts-run-%s.pt" % (mode, dataset,
                                                                 loss_name,
                                                                 noise_type,
                                                                 str(int(noise_rate
                                                                 * 10)),
                                                                 str(run)))
            print("Model weights updated...\n")

        print("Epoch: {}, lr: {}, loss_train: {}, loss_val: {}, loss_test: {:.3f}, acc_train: {},\
              acc_val: {}, acc_test: {:.3f}\n".format(epoch, optimizer.param_groups[0]['lr'],
                                                      loss_train, loss_val, loss_test, 
                                                      acc_train, acc_val, acc_test))


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
    with open(res_path +
              "%s-%s-%s-%s-nr-0%s-run-%s.pickle" % (mode, dataset, loss_name,
                                                    noise_type, str(int(noise_rate * 10)),
                                                    str(run)), 'wb') as f:
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

    print("Pickle file saved: " + res_path +
          "%s-%s-%s-%s-nr-0%s-run-%s.pickle" % (mode, dataset, loss_name, noise_type,
                                                str(int(noise_rate * 10)), str(run)), "\n")