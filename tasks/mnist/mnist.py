import argparse

import json
import os

import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import metrics
import tasks.mnist.models as models

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

normalize = transforms.Normalize((0.1307,), (0.3081,))

transforms = transforms.Compose([
        transforms.ToTensor(),
        normalize
        ])

train_transforms = transforms
validation_transforms = transforms
preprocess = transforms

# todo: use @property

def train_dataset(data_dir):
    return datasets.MNIST(data_dir, train=True, download=True,
                          transform=train_transforms)

def validation_dataset(data_dir):
    return datasets.MNIST(data_dir, train=False,
                          transform=validation_transforms)

def default_epochs():
    return 10

def default_initial_lr():
    return 1

def default_lr_scheduler(optimizer, num_epochs, steps_per_epoch, start_epoch=0):
    return torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.7, last_epoch=start_epoch - 1)

def default_optimizer(model, lr, momentum, weight_decay):
    return torch.optim.Adadelta(model.parameters(), lr=lr)

def to_device(batch, device, gpu_id):
    (images, target) = batch
    if gpu_id is not None:
        images = images.cuda(gpu_id, non_blocking=True)
    if device.startswith("cuda"):
        target = target.cuda(gpu_id, non_blocking=True)
    return (images, target)

def get_input(batch):
    (images, _) = batch
    return images, {}

def get_target(batch):
    (_, target) = batch
    return target

def default_criterion():
    return torch.nn.CrossEntropyLoss()

def get_loss(output, batch, criterion):
    (_, target) = batch
    return criterion(output, target)

def default_metrics():
    return topk(1,5)

def get_metrics(output, target, **kwargs):
    metrics_dict = dict()
    if "topk" in kwargs:
        acc1, acc5 = metrics.accuracy(output, target, kwargs["topk"])
        metrics_dict["acc1"] = acc1
        metrics_dict["acc5"] = acc5
    return metrics_dict

idx2label = [str(k) for k in range(10)]