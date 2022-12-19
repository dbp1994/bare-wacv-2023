from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image
import json
import torch

"""
Code taken from 

Meta MLNT: https://github.com/LiJunnan1992/MLNT/blob/40f163d42b05c990b499a6d1a5539150986ab370/dataloader.py
"""

class clothing_dataset(Dataset): 
    def __init__(self, transform, mode, batch_size=32, num_batches = 1000, num_class=14): 
        
        self.train_imgs = []
        self.test_imgs = []
        self.val_imgs = []
        self.train_labels = {}
        self.test_labels = {}
        self.val_labels = {}
        self.transform = transform
        self.mode = mode
        self.batch_size = batch_size
        self.num_batches = num_batches

        with open('./data/clothing1M/noisy_label_kv.txt','r') as f:
            lines = f.read().splitlines()
        for l in lines:
            entry = l.split()           
            img_path = './data/clothing1M/'+entry[0][7:]
            self.train_labels[img_path] = int(entry[1])

        with open('./data/clothing1M/clean_label_kv.txt','r') as f:
            lines = f.read().splitlines()
        for l in lines:
            entry = l.split()           
            img_path = './data/clothing1M/'+entry[0][7:]
            self.test_labels[img_path] = int(entry[1])


        with open('./data/clothing1M/noisy_train_key_list.txt','r') as f:
            lines = f.read().splitlines()
        train_imgs = []
        for l in lines:
            img_path = './data/clothing1M/'+l[7:]
            train_imgs.append(img_path)
            # self.train_imgs.append(img_path)
        random.shuffle(train_imgs)
        class_num = torch.zeros(num_class)
        for impath in train_imgs:
            label = self.train_labels[impath]
            if class_num[label] <((self.num_batches * self.batch_size)/14) and len(self.train_imgs) < (self.num_batches * self.batch_size):
                self.train_imgs.append(impath)
                class_num[label] += 1

        with open('./data/clothing1M/clean_test_key_list.txt','r') as f:
            lines = f.read().splitlines()
        for l in lines:
            img_path = './data/clothing1M/'+l[7:]
            self.test_imgs.append(img_path)

        with open('./data/clothing1M/clean_val_key_list.txt','r') as f:
            lines = f.read().splitlines()
        for l in lines:
            img_path = './data/clothing1M/'+l[7:]
            self.val_imgs.append(img_path)
            
  

            
    def __getitem__(self, index):  
        if self.mode=='train':
            img_path = self.train_imgs[index]
            target = self.train_labels[img_path]     
        elif self.mode=='test':
            img_path = self.test_imgs[index]
            target = self.test_labels[img_path]
        elif self.mode=='val':
            img_path = self.val_imgs[index]
            target = self.test_labels[img_path]            
        image = Image.open(img_path).convert('RGB')    
        img = self.transform(image)
        return img, target
    
    def __len__(self):
        if self.mode=='train':
            return len(self.train_imgs)
        elif self.mode=='test':
            return len(self.test_imgs)      
        elif self.mode=='val':
            return len(self.val_imgs)

        
class clothing_dataloader():  
    def __init__(self, batch_size, shuffle):    
        self.batch_size = batch_size
        # self.num_workers = num_workers
        # self.num_batches = num_batches
        self.shuffle = shuffle
    
    def run(self):
        # transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
        self.transform_train = transforms.Compose([
                transforms.Resize(256),
                # transforms.RandomCrop(224),
                transforms.CenterCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),                
                transforms.Normalize((0.6959, 0.6537, 0.6371),(0.3113, 0.3192, 0.3214)),
            ]) 
        self.transform_test = transforms.Compose([
                transforms.Resize(256),
                # transforms.RandomCrop(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.6959, 0.6537, 0.6371),(0.3113, 0.3192, 0.3214)),
            ])
        
        train_dataset = clothing_dataset(transform=self.transform_train, mode='train', batch_size=self.batch_size)
        test_dataset = clothing_dataset(transform=self.transform_test, mode='test', batch_size=self.batch_size)
        val_dataset = clothing_dataset(transform=self.transform_test, mode='val', batch_size=self.batch_size)
        
        train_loader = DataLoader(
            dataset=train_dataset, 
            batch_size=self.batch_size,
            shuffle=self.shuffle)
        test_loader = DataLoader(
            dataset=test_dataset, 
            batch_size=self.batch_size,
            shuffle=False)
        val_loader = DataLoader(
            dataset=val_dataset, 
            batch_size=self.batch_size,
            shuffle=False)
        return train_loader, val_loader, test_loader