import argparse
import json
import os

import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import lenet as models
model_names = ['lenet']

normalize = transforms.Normalize((0.1307,), (0.3081,))

transforms = transforms.Compose([
        transforms.ToTensor(),
        normalize
        ])

train_transforms = transforms
validation_transforms = transforms
preprocess = transforms

def train_dataset(data_dir):
    return datasets.MNIST(data_dir, train=True, download=True,
                          transform=train_transforms)

def validation_dataset(data_dir):
    return datasets.MNIST(data_dir, train=False,
                          transform=validation_transforms)

def default_initial_lr():
    return 1

def default_lr_scheduler(optimizer, start_epoch):
    return torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.7, last_epoch=start_epoch - 1)

def default_optimizer(model, lr, momentum, weight_decay):
    return torch.optim.Adadelta(model.parameters(), lr=lr)

idx2label = [str(k) for k in range(10)]