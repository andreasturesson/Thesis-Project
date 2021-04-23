import torch
import numpy as np
from torch.utils.data import (TensorDataset, DataLoader, RandomSampler, SequentialSampler)


def load_trainset(train_path_X, train_path_y, disc_train_path_X, disc_train_path_y):
    # Load datasets from disc
    training_data_X = torch.tensor(np.load(
        train_path_X, allow_pickle=True))
    training_data_y = torch.tensor(np.load(
        train_path_y, allow_pickle=True).astype(int))

    disc_eval_data_X = torch.tensor(np.load(
        disc_train_path_X, allow_pickle=True))
    disc_eval_data_y = torch.tensor(np.load(
        disc_train_path_y, allow_pickle=True).astype(int))

    # Validation/train data split
    VAL_PCT = 0.1
    val_size = int(len(training_data_X)*VAL_PCT)
    disc_val_size = int(len(disc_eval_data_X)*VAL_PCT)

    # Split train data
    train_inputs = training_data_X[:-val_size]
    train_labels = training_data_y[:-val_size]

    val_inputs = training_data_X[-val_size:]
    val_labels = training_data_y[-val_size:]

    disc_val_inputs = disc_eval_data_X[-disc_val_size:]
    disc_val_labels = disc_eval_data_y[-disc_val_size:]

    return train_inputs, train_labels, val_inputs, val_labels, disc_val_inputs, disc_val_labels


def load_holdout_set(test_path_X, test_path_y):
    test_data_X = torch.tensor(np.load(
        test_path_X, allow_pickle=True))
    test_data_y = torch.tensor(np.load(
        test_path_y, allow_pickle=True).astype(int))
    return test_data_X, test_data_y


def data_loader(train_inputs, train_labels, val_inputs, val_labels, disc_val_inputs, disc_val_labels, batch_size=64):
    train_data = TensorDataset(train_inputs, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    val_data = TensorDataset(val_inputs, val_labels)
    val_sampler = RandomSampler(val_data)
    val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)

    disc_val_data = TensorDataset(disc_val_inputs, disc_val_labels)
    disc_val_sampler = SequentialSampler(disc_val_data)
    disc_val_dataloader = DataLoader(disc_val_data, sampler=disc_val_sampler, batch_size=batch_size)

    return train_dataloader, val_dataloader, disc_val_dataloader


def data_loader_holdout(test_inputs, test_labels, batch_size=64):
    test_data = TensorDataset(test_inputs, test_labels)
    sampler = SequentialSampler(test_data)
    dataloader = DataLoader(test_data, sampler=sampler, batch_size=batch_size)

    return dataloader


def set_seed(seed_value=42):
    """Set seed for reproducibility."""

    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)