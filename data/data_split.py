from glob import glob
import pdb
import json
import random
import numpy as np
import os

def split_prostate():
    if not os.path.exists('data_split/Prostate/'):
        os.makedirs('data_split/Prostate/')

    for i in range(6):
        data_list = glob('../../Dataset/Prostate_npy/client{}/data_npy/*'.format(i+1))
        np.random.shuffle(data_list)   # prostate -> case
        data_len = len(data_list)   # prostate -> num of cases

        test_len = int(np.ceil(0.2*data_len))
        test_list = data_list[:test_len]
        train_val_list = data_list[test_len:]

        val_len = int(np.ceil(0.1*data_len))
        val_list = train_val_list[:val_len]
        train_list = train_val_list[val_len:]
        
        train_slice_list = []
        for train_case in train_list:
            train_slice_list.extend(glob('{}/*'.format(train_case)))
        print(len(train_slice_list))
        with open("data_split/Prostate/client{}_train.txt".format(i+1), "w") as f: 
            json.dump(train_slice_list, f)

        val_slice_list = []
        for val_case in val_list:
            val_slice_list.extend(glob('{}/*'.format(val_case)))
        print(len(val_slice_list))
        with open("data_split/Prostate/client{}_val.txt".format(i+1), "w") as f: 
            json.dump(val_slice_list, f)

        test_slice_list = []
        for test_case in test_list:
            test_slice_list.extend(glob('{}/*'.format(test_case)))
        print(len(test_slice_list))
        with open("data_split/Prostate/client{}_test.txt".format(i+1), "w") as f: 
            json.dump(test_slice_list, f)

# split_prostate()

def split_dataset(dataset, client_num):
    if not os.path.exists('data_split/{}'.format(dataset)):
        os.makedirs('data_split/{}'.format(dataset))

    for i in range(client_num):   
        data_list = glob('../../Dataset/{}_npy/client{}/data_npy/*'.format(dataset, i+1))
        np.random.shuffle(data_list)

        data_len = len(data_list)
        test_len = int(0.2*data_len)
        test_list = data_list[:test_len]

        train_val_list = data_list[test_len:]
        val_len = int(0.1*data_len)
        val_list = train_val_list[:val_len]
        train_list = train_val_list[val_len:]
        
        print(len(test_list), len(val_list), len(train_list))

        with open("data_split/{}/client{}_train.txt".format(i+1), "w") as f: json.dump(train_list, f)
        with open("data_split/{}/client{}_val.txt".format(i+1), "w") as f: json.dump(val_list, f)
        with open("data_split/{}/client{}_test.txt".format(i+1), "w") as f: json.dump(test_list, f)

split_dataset(dataset='Polyp', client_num=4)