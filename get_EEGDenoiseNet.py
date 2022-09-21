import torch
import torch.optim as optim
import torch.utils.data as Data
import torch.nn as nn
import os
import numpy as np


def standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma


BATCH_SIZE = 1000

def get_EEGDenoiseNet_trainPair(noise_type):
    if(noise_type == 'EOG'):
        artifact = np.load('/data/Liulei/preprocess/data/EOG_all_epochs.npy')
    elif(noise_type == 'EMG'):
        artifact = np.load('/data/Liulei/preprocess/data/EMG_all_epochs.npy')
    raw_eeg = np.load('/data/Liulei/preprocess/DeepSeparator-main/data/EOGtrain_input.npy')
    clean_eeg = np.load('/data/Liulei/preprocess/DeepSeparator-main/data/EOGtrain_output.npy')
    return clean_eeg,raw_eeg

def get_EEGDenoiseNet(noise_type):
    if(noise_type == 'EOG'):
        artifact = np.load('/data/Liulei/preprocess/data/EOG_all_epochs.npy')
    elif(noise_type == 'EMG'):
        artifact = np.load('/data/Liulei/preprocess/data/EMG_all_epochs.npy')
    raw_eeg = np.load('/data/Liulei/preprocess/DeepSeparator-main/data/EOGtrain_input.npy')
    clean_eeg = np.load('/data/Liulei/preprocess/DeepSeparator-main/data/EOGtrain_output.npy')
    
    test_input = np.load('/data/Liulei/preprocess/DeepSeparator-main/data/EOG_EEG_test_input.npy')
    test_output = np.load('/data/Liulei/preprocess/DeepSeparator-main/data/EOG_EEG_test_output.npy')

    train_input = torch.from_numpy(raw_eeg)
    train_output = torch.from_numpy(clean_eeg)

    train_torch_dataset = Data.TensorDataset(train_input, train_output)

    train_loader = Data.DataLoader(
        dataset=train_torch_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    test_input = torch.from_numpy(test_input)
    test_output = torch.from_numpy(test_output)

    test_torch_dataset = Data.TensorDataset(test_input, test_output)

    test_loader = Data.DataLoader(
        dataset=test_torch_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    return train_loader, test_loader
