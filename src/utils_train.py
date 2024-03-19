"""Script with training ancillar functions."""

import math
import yaml
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler as schedulers
from matplotlib import pyplot as plt


# Use an LLM for the docs comment asking to
#   write a documentation comment for the following function considering the geranel idea, inputs and outputs using the style of the first one
# then copy the answer in the script.


def train_one_epoch(model, train_loader, loss_fn, optimizer, scheduler=None):
    """
    Trains a model for one epoch. This function iterates over the training data loader, 
    feeding batches of data to the model, calculating the loss between predictions and 
    true labels, and updating the model's weights to improve performance. It tracks the 
    total loss across the epoch and returns the average loss.

    Args:
        model: The model to be trained (typically a subclass of nn.Module).
        train_loader: A DataLoader containing training data (input-label pairs).
        loss_fn: A function that computes the loss between predictions and labels.
        optimizer: An optimizer that updates the model's weights based on the loss.
        scheduler (optional): A learning rate scheduler to adjust learning rate.

    Returns:
        float: The average loss of the model over the entire epoch.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    running_loss = 0.

    for x,y in train_loader:
        # Every data instance is an input + label pair
        x,y = x.to(device), y.to(device)

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        y_hat = model(x)

        # Compute the loss and its gradients
        loss = loss_fn(y_hat, y)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()

    if scheduler is not None:
        scheduler.step()
    return running_loss/len(train_loader)


def load_yaml_as_dict(filename):
    """
    Loads a YAML file and returns the data as a Python dictionary.

    Args:
        filename (str): The path to the YAML file.

    Returns:
        dict: The data from the YAML file as a dictionary.
    """
    try:
        with open(filename, 'r', encoding="utf-8") as file: # added encoding
            data = yaml.safe_load(file)
            return data
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
    return None


def select_loss_function(loss_name):
    """
    Selects the appropriate PyTorch loss function based on a given name.

    Args:
        loss_name (str): The name of the desired loss function.

    Returns:
        nn.Module: The selected PyTorch loss function object.

    Raises:
        ValueError: If the provided loss name is not recognized.
    """

    loss_function_map = {
        "mse": nn.MSELoss(),
        "cross_entropy": nn.CrossEntropyLoss(),
        "bce_with_logits": nn.BCEWithLogitsLoss(),
        "l1": nn.L1Loss()
    }

    try:
        return loss_function_map[loss_name.lower()]  # Ensure case-insensitive lookup
    except KeyError as e:
        raise ValueError(f"Invalid loss function name: {loss_name}") from e


def select_optimizer(model, optimizer_name, learning_rate=0.01, momentum=0.9):
    """
    Selects the appropriate PyTorch optimizer based on a given name.

    Args:
        optimizer_name (str): The name of the desired optimizer.
        learning_rate (float, optional): The learning rate to use with the optimizer. Defaults to 0.01.
        momentum (float, optional): The momentum parameter to use with certain optimizers (e.g., SGD). Defaults to 0.9.

    Returns:
        optim.Optimizer: The selected PyTorch optimizer object.

    Raises:
        ValueError: If the provided optimizer name is not recognized.
    """

    optimizer_map = {
        "sgd": optim.SGD(params=model.parameters(), lr=learning_rate, momentum=momentum),
        "adam": optim.Adam(params=model.parameters(), lr=learning_rate),
        "adamw": optim.AdamW(params=model.parameters(), lr=learning_rate),
        "rmsprop": optim.RMSprop(params=model.parameters(), lr=learning_rate),
    }

    try:
        return optimizer_map[optimizer_name.lower()]  # Ensure case-insensitive lookup
    except KeyError as e:
        raise ValueError(f"Invalid optimizer name: {optimizer_name}") from e


def select_scheduler(scheduler_name, optimizer, last_epoch=-1, **kwargs):
    """
    Selects the appropriate PyTorch learning rate scheduler based on a given name.

    Args:
        scheduler_name (str): The name of the desired scheduler.
        optimizer (optim.Optimizer): The optimizer object to use with the scheduler.
        last_epoch (int, optional): The epoch index from the previous training step. Defaults to -1.
        **kwargs: Additional keyword arguments specific to the chosen scheduler.

    Returns:
        schedulers._LRScheduler: The selected PyTorch learning rate scheduler object.

    Raises:
        ValueError: If the provided scheduler name is not recognized.
    """

    scheduler_map = {
        "reduce_on_plateau": schedulers.ReduceLROnPlateau(optimizer=optimizer, **kwargs),
        "cosine_annealing_lr": schedulers.CosineAnnealingLR(optimizer=optimizer, last_epoch=last_epoch, **kwargs),
        "step_lr": schedulers.StepLR(optimizer=optimizer, **kwargs),
    }

    try:
        return scheduler_map[scheduler_name.lower()]  # Ensure case-insensitive lookup
    except KeyError as e:
        raise ValueError(f"Invalid scheduler name: {scheduler_name}") from e
