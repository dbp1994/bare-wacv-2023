# Adaptive Sample Selection for Robust Learning under Label Noise (IEEE/CVF WACV 2023)

## Installation

Make a copy of this repo (e.g. with git clone), ```cd``` into the root folder of the repo, and run:

> pip install -e .

## Organization

This project is organized into folders:
- ```data``` should contain all the dataset files
- ```scripts``` contain scripts for all the algorithms
- ```results``` should contain all the output pickle files, checkpoints, etc.

## Running the experiments

```cd``` into the scripts folder and run ```algo.py``` where ```algo```  $\in $ {BARE, MR, MN, CoT, CoT+, CL, CCE}
