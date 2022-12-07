# Adaptive Sample Selection for Robust Learning under Label Noise (IEEE/CVF WACV 2023) [`Paper`](https://arxiv.org/abs/2106.15292)

## Installation

Make a copy of this repo (e.g. with git clone), ```cd``` into the root folder of the repo, and run:

> pip install -e .

## Requirements
- PyTorch >= 1.3
- Python >= 3.7
- tqdm, numpy-indexed, etc (which can be easily installed via pip)

## Organization

This project is organized into folders:
- ```data``` should contain all the dataset files
- ```scripts``` contain scripts for all the algorithms
- ```results``` should contain all the output pickle files, checkpoints, etc.

## Running the experiments

```cd``` into the scripts folder and run ```algo.py``` where ```algo```  is the algorithm used for training.

For the sake of completeness, filenames for each of the algorithms (names as per convention in the paper) are as follows:
1. BARE - ```batch_rewgt.py```
2. MR - ```meta_ren.py```
3. MN - ```meta_net.py```
4. CoT - ```coteaching.py```
5. CoT+ - ```coteaching.py```
6. CL - ```curr_loss.py```
7. CCE - ```risk_min_cce.py```

For example, if you want to use BARE, then one such command could look like this:

```
python batch_rewgt.py --dataset mnist --noise_rate 0.4 --noise_type sym --loss_name cce --data_aug 0 --batch_size 128 --num_epoch 200 --num_runs 5 
```

This will train the neural network with CCE loss on un-augmented MNIST dataset which will be corrupted with 40% symmetric label noise. The training will be carried for a batch-size of 128, 200 epochs and a total of 5 runs.


## Reference

The following citation can be used:

```
@inproceedings{bare_wacv_2023,
  title={Adaptive Sample Selection for Robust Learning under Label Noise,
  author={Patel, Deep and Sastry, P S},
  booktitle={WACV},
  year={2023}
  }
```
