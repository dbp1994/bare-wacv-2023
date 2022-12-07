import time

import numpy as np
from sklearn import model_selection
import torchvision
from torchvision import transforms

import numpy_indexed as npi
from add_noise import (noisify_with_P, 
                       noisify_mnist_asymmetric,
                       noisify_cifar10_asymmetric)

# set seed for reproducibility
np.random.seed(3459)

def numpy_to_categorical(y, num_classes=None, dtype='float32'):
    """
    Taken from Keras repo
    https://github.com/keras-team/keras/blob/master/keras/utils/np_utils.py#L9-L37
    """
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical

def read_data(noise_type, noise_rate, dataset, data_aug=False, mode="risk_min"):
    if dataset == "mnist":
        num_class = 10
        if not data_aug:
            dat_train = torchvision.datasets.MNIST('./data', train=True, download=True,
                                                   transform=transforms.Compose(
                                                       [transforms.ToTensor(),
                                                        transforms.Normalize((0.1307, ),
                                                                             (0.3081, ))]))

            dat_test = torchvision.datasets.MNIST('./data', train=False, download=True,
                                                  transform=transforms.Compose(
                                                      [transforms.ToTensor(),
                                                       transforms.Normalize((0.1307, ),
                                                                            (0.3081, ))]))
            print("\nDATA AUGMENTATION DISABLED...\n")
        else:
            raise NotImplementedError("Data augmentation not implemented.\n")


        X_temp = (dat_train.data).numpy()
        y_temp = (dat_train.targets).numpy()

        X_test = (dat_test.data).numpy()
        y_test = (dat_test.targets).numpy()
        
        feat_size = 28 * 28

    elif dataset == "cifar10":
        num_class = 10

        # data_aug = True

        if not data_aug:
            dat_train = torchvision.datasets.CIFAR10('./data', train=True, download=True,
                                                     transform=transforms.Compose(
                                                         [transforms.ToTensor(),
                                                          transforms.Normalize((0.4914, 0.4822,
                                                                                0.4465),
                                                                               (0.2023, 0.1994,
                                                                                0.2010))]))

            dat_test = torchvision.datasets.CIFAR10('./data', train=False, download=True,
                                                    transform=transforms.Compose(
                                                        [transforms.ToTensor(),
                                                         transforms.Normalize((0.4914, 0.4822,
                                                                               0.4465),
                                                                              (0.2023, 0.1994,
                                                                               0.2010))]))

            print("\nDATA AUGMENTATION DISABLED...\n")

        else:
            dat_train = torchvision.datasets.CIFAR10('./data', train=True, download=True,
                                                     transform=transforms.Compose(
                                                         [transforms.RandomCrop(32, padding=4),
                                                          transforms.RandomHorizontalFlip(),
                                                          transforms.ToTensor(),
                                                          transforms.Normalize((0.4914,
                                                                                0.4822, 0.4465),
                                                                               (0.2023, 0.1994,
                                                                                0.2010))]))

            # dat_train = torchvision.datasets.CIFAR10('./data', train=True, download=True,
            #                                          transform=transforms.Compose(
            #                                              [transforms.CenterCrop(28),
            #                                               transforms.RandomHorizontalFlip(),
            #                                               transforms.ToTensor(),
            #                                               transforms.Normalize((0.4914,
            #                                                                     0.4822, 0.4465),
            #                                                                    (0.2023, 0.1994,
            #                                                                     0.2010))]))

            dat_test = torchvision.datasets.CIFAR10('./data', train=False, download=True,
                                                    transform=transforms.Compose(
                                                        [transforms.ToTensor(),
                                                         transforms.Normalize((0.4914,
                                                                               0.4822, 0.4465),
                                                                              (0.2023, 0.1994,
                                                                               0.2010))]))
            print("\nDATA AUGMENTATION ENABLED...\n")

        X_temp = dat_train.data
        y_temp = np.asarray(dat_train.targets)
        X_test = dat_test.data
        y_test = np.asarray(dat_test.targets)

        feat_size = 3 * 32 * 32
    else:
        raise SystemExit("Dataset not supported.\n")


    #num_class = np.max(y_temp) + 1

    """
    Conv layers in PyTorch expect the data to be 4-D wherein #channels
    is the additional dimension
    i.e. (n_samples, channels, height, width)
    """
    if dataset == "mnist":
        X_temp = np.expand_dims(X_temp, 1)  # if numpy array
        # X_temp = X_temp.unsqueeze(1)  # if torch tensor

        X_test = np.expand_dims(X_test, 1)  # if numpy array
        # X_test = X_test.unsqueeze(1)  # if torch tensor

    elif dataset in ["cifar10", "cifar100"]:
        X_temp = X_temp.transpose(0, 3, 1, 2)   # if numpy array
        # X_temp = X_temp.permute(0, 3, 1, 2)   # if torch tensor

        X_test = X_test.transpose(0, 3, 1, 2)   # if numpy array
        # X_test = X_test.permute(0, 3, 1, 2)   # if torch tensor
    """
    Add Label Noise
    """
    if mode in ["meta_ren", "meta_net"]:
        num_val_clean = 1000
        idx_val_clean = []

        y_temp = y_temp.astype(np.int32)

        # Pick equal no. clean samples from each class for val. set
        if dataset in ["mnist", "cifar10"]:
            X_train, X_val, y_train, y_val = model_selection.train_test_split(X_temp, y_temp,
                                                                              test_size=0.2,
                                                                              random_state=42)

            num_class = 10

            for i in range(num_class):
                idx_cls_tmp = np.where(y_val == i)[0]
                rng = np.random.default_rng()
                tmp_pick = rng.choice(idx_cls_tmp.shape[0], 
                                      size=num_val_clean//num_class,
                                      replace=False)
                # tmp_pick = np.random.randint(low=0, high=idx_cls_tmp.shape[0],
                #                              size=(num_val_clean//num_class,))
                if i == 0:
                    idx_val_clean = idx_cls_tmp[tmp_pick]
                else:
                    idx_val_clean = np.concatenate((idx_val_clean,
                                                    idx_cls_tmp[tmp_pick]), axis=0)
        else:
            raise SystemExit("Dataset not supported.\n")

        # Train. and Val. data are from the same set, so separate them out
        X_val = (X_val[idx_val_clean, :, :, :]).copy()
        y_val = (y_val[idx_val_clean]).copy()

        if noise_rate > 0.:
            if noise_type == 'sym':
                y_train_noisy, P = noisify_with_P(y_train, num_class, noise_rate,
                                                  random_state=42)
            elif noise_type == 'cc':
                if dataset == "mnist":
                    y_train_noisy, P = noisify_mnist_asymmetric(y_train, noise_rate,
                                                                random_state=42)
                elif dataset == "cifar10":
                    y_train_noisy, P = noisify_cifar10_asymmetric(y_train, noise_rate,
                                                                  random_state=42)
                else:
                    raise NotImplementedError("Not implemented for this datasets.")
            else:
                raise SystemExit("Noise type not supported.")
        else:
            y_train_noisy = y_train
    else:
        num_val_clean = 1000
        idx_val_clean = []

        y_temp = y_temp.astype(np.int32)

        # Pick equal no. clean samples from each class for val. set
        if dataset in ["mnist", "cifar10"]:
            for i in range(num_class):
                idx_cls_tmp = np.where(y_val == i)[0]
                rng = np.random.default_rng()
                tmp_pick = rng.choice(idx_cls_tmp.shape[0],
                                      size=num_val_clean//num_class,
                                      replace=False)
                # tmp_pick = np.random.randint(low=0,
                #                              high=idx_cls_tmp.shape[0],
                #                              size=(num_val_clean//num_class,))
                if i == 0:
                    idx_val_clean = idx_cls_tmp[tmp_pick]
                else:
                    idx_val_clean = np.concatenate((idx_val_clean,
                                                    idx_cls_tmp[tmp_pick]), axis=0)

        else:
            raise SystemExit("Dataset not supported.\n")

        # Train. and Val. data are from the same set, so separate them out
        X_val = (X_val[idx_val_clean, :, :, :]).copy()
        y_val = (y_val[idx_val_clean]).copy()
        y_val_shape = y_val.shape[0]

        # print("===========================\n")
        # print(X_train.shape, X_val.shape, y_train.shape, y_val.shape)
        # print("\n===========================\n")

        y_temp1 = np.concatenate((y_train, y_val))

        if noise_rate > 0.:
            if noise_type == 'sym':
                y_temp_noisy, P = noisify_with_P(y_temp1,
                                                 num_class,
                                                 noise_rate,
                                                 random_state=42)
            elif noise_type == "cc":
                if dataset == "mnist":
                    y_temp_noisy, P = noisify_mnist_asymmetric(y_temp1,
                                                               noise_rate,
                                                               random_state=42)
                elif dataset == "cifar10":
                    y_temp_noisy, P = noisify_cifar10_asymmetric(y_temp1,
                                                                 noise_rate,
                                                                 random_state=42)
                    # y_temp_n, _ = noisify_cifar10_asymmetric(y_temp,
                    #                                          noise_rate,
                    #                                          random_state=42)
                    # y_temp_noisy, P = noisify_with_P(y_temp_n,
                    #                                  num_class,
                    #                                  noise_rate=0.9,
                    #                                  random_state=42)
                    # actual_noise = (y_temp_noisy != y_temp).mean()
                    # assert actual_noise > 0.0
                    # print('Actual noise %.2f' % actual_noise)
                else:
                    raise NotImplementedError("Not implemented for this dataset.")
            else:
                raise SystemExit("Noise type not supported.")
        else:
            y_temp_noisy = y_temp1
        
        y_train_noisy = y_temp_noisy[:-y_val_shape]
        y_val = y_temp_noisy[-y_val_shape:]
        
    """
    Since the data has been shuffled during 'sklearn.model_selection',
    we wil keep track of the indices so as to infer clean and 
    noisily-labelled samples
    """

    if noise_rate > 0.:
        dat_noisy = np.concatenate((X_train, X_val), axis=0)
        print(dat_noisy.shape)
        if dataset == "mnist":
            X = X_temp.reshape((X_temp.shape[0], 784))
            dat_noisy = dat_noisy.reshape((dat_noisy.shape[0], 784))
        elif dataset == "cifar10":
            X = X_temp.reshape((X_temp.shape[0], 1024*3))
            dat_noisy = dat_noisy.reshape((dat_noisy.shape[0], 1024*3))
        else:
            raise NotImplementedError("Not implemented for this dataset.")

        print(dat_noisy.shape)
        print(X.shape)

        idx = []
        idx_train_clean = []
        idx_train_noisy = []
        start = time.perf_counter()
        idx = npi.indices(X, dat_noisy)
        stop = time.perf_counter()
        time_taken = stop - start
        print("search time: ", time_taken, "\n")

        print("idx[5]: ", idx[5])
        print((X[idx[5]] == dat_noisy[5]).all())

        idx_train = idx[0:X_train.shape[0]]
        idx_val = idx[X_train.shape[0]:]
        print(len(idx_train))
        print(len(idx_val))
        print("y_train: {}".format(y_train_noisy.shape))


        idx_tr_clean_ref = np.where(y_temp[idx_train] == y_train_noisy)[0]
        idx_tr_noisy_ref = np.where(y_temp[idx_train] != y_train_noisy)[0]

        if dataset == "mnist":
            idx_train_clean = npi.indices(X, X_train.reshape(-1, 784)[idx_tr_clean_ref, :])
            idx_train_noisy = npi.indices(X, X_train.reshape(-1, 784)[idx_tr_noisy_ref, :])
        elif dataset == "cifar10":
            idx_train_clean = npi.indices(X, X_train.reshape(-1, 1024*3)[idx_tr_clean_ref, :])
            idx_train_noisy = npi.indices(X, X_train.reshape(-1, 1024*3)[idx_tr_noisy_ref, :])
        else:
            raise NotImplementedError("Not implemented for this dataset.")

        
        ids = (idx, idx_train, idx_val, idx_tr_clean_ref,
               idx_tr_noisy_ref, idx_train_clean,
               idx_train_noisy, P)
        
    else:
        idx = np.asarray(list(range(X_temp.shape[0])))
        idx_train = np.asarray(list(range(X_train.shape[0])))
        idx_val = np.asarray(list(range(X_val.shape[0])))
        idx_train_clean = []
        idx_train_noisy = []

        ids = (idx, idx_train, idx_val, idx_train_clean,
               idx_train_noisy)
    


    dat = (X_temp, y_temp, X_train, y_train_noisy,
           X_val, y_val, X_test, y_test)
    print(f"X_temp: {X_temp.shape}, X_test: {X_test.shape}\n")    
    return dat, ids
