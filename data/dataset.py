import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms as T
from scipy import ndimage
import albumentations as A
from PIL import Image
import cv2
from scipy.ndimage import zoom
import os
from glob import glob
import random
import numpy as np
import json
import pdb
import pandas as pd
import pickle
import math
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision
import skimage


class Polyp(Dataset):
    def __init__(self, fl_method, client_idx=None, mode='train', transform=None):
        assert mode in ['train', 'val', 'test']

        self.num_classes = 2
        self.fl_method = fl_method

        self.client_name = ['client1', 'client2', 'client3', 'client4']
        self.client_idx = client_idx    # obtain the dataset of client_name[client_idx]

        self.mode = mode
        self.transform = transform

        self.data_list = []

        with open("data/data_split/Polyp/{}_{}.txt".format(self.client_name[self.client_idx], mode), "r") as f: 
            self.data_list = json.load(f)

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx: int):
        data_path = self.data_list[idx]
        data = np.load(data_path)

        image = data[..., 0:3]
        label = data[..., 3:]   

        sample = {'image':image, 'label':label}
        if self.transform is not None:
            sample = self.transform(sample) 

        return idx, sample


class RandomFlip(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, sample):
        image = sample['image']
        label = sample['label']

        if np.random.uniform() > self.p:
            axis = np.random.randint(0, 2)
            image = np.flip(image, axis=axis).copy()
            label = np.flip(label, axis=axis).copy()
            
        return {'image': image, 'label': label}


class ToTensor(object):
    def __call__(self, sample):
        image = sample['image'].transpose(2, 0, 1).astype(np.float32)
        label = sample['label'].transpose(2, 0, 1)

        return {'image': torch.from_numpy(image.copy()/image.max()), 'label': torch.from_numpy(label.copy()).long()}


def generate_dataset(dataset, fl_method, client_idx):
    if dataset == 'Polyp':
        from data.dataset import Polyp as Med_Dataset
        train_transform = T.Compose([RandomFlip(p=0.5),ToTensor()]) 
        test_transform = T.Compose([ToTensor()])

    data_train = Med_Dataset(fl_method=fl_method, 
                                client_idx=client_idx,
                                mode='train',
                                transform=train_transform)
    
    data_val = Med_Dataset(fl_method=fl_method,
                            client_idx=client_idx,
                            mode='val',
                            transform=test_transform)

    data_test = Med_Dataset(fl_method=fl_method,
                                client_idx=client_idx,
                                mode='test',
                                transform=test_transform)
                                
    return data_train, data_val, data_test