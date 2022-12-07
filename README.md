# Adaptive Sample Selection for Robust Learning under Label Noise (IEEE/CVF WACV 2023)

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

For example,

> python bare.py -dat MN

## Reference

The following citation can be used:

```@inproceedings{bare_wacv_2023,
  title={Adaptive Sample Selection for Robust Learning under Label Noise,
  author={Patel, Deep and Sastry, P S},
  booktitle={WACV},
  year={2023}
  }
```
