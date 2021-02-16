# code is taken from https://chriskhanhtran.github.io/posts/cnn-sentence-classification/

import torch
import numpy as np
from torch.utils.data import (TensorDataset, DataLoader, RandomSampler)

def load_trainset(path_X, path_y):
    # Load datasets from disc
    training_data_X = torch.tensor(np.load(
        path_X, allow_pickle=True))
    training_data_y = torch.tensor(np.load(
        path_y, allow_pickle=True).astype(int))

    # Turn the labels from one hot to scalar: [0,1] or [1,0] to [0] or [1] (not haha)
    # Loss function CrossEntropyLoss only takes scalar
    # Remove line below if loss function require one hot

    # Validation/train data split
    VAL_PCT = 0.1
    val_size = int(len(training_data_X)*VAL_PCT)

    # Split train data
    train_inputs = training_data_X[:-val_size]
    train_labels = training_data_y[:-val_size]

    val_inputs = training_data_X[-val_size:]
    val_labels = training_data_y[-val_size:]

    return train_inputs, train_labels, val_inputs, val_labels


def data_loader(train_inputs, train_labels, val_inputs, val_labels, batch_size=50):
    train_data = TensorDataset(train_inputs, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    val_data = TensorDataset(val_inputs, val_labels)
    val_sampler = RandomSampler(val_data)
    val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)

    return train_dataloader, val_dataloader