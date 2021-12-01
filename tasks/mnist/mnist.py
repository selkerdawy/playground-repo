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
# todo: create Task class so that tasks override it?

# todo: add default_model
# todo: add default_batch_size

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

def default_loss_fn():
    return torch.nn.NLLLoss()

def get_loss(output, batch, loss_fn):
    (_, target) = batch
    return loss_fn(output, target)

def default_metrics_fn():
    return metrics.accuracy(topk=(1,))

def get_metrics(output, target, metrics_fn):
    return metrics_fn(output, target)

idx2label = [str(k) for k in range(10)]