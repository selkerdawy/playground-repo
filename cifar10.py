import argparse

import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import cifar10_models as models

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

normalize = transforms.Normalize(
    mean=[0.4914, 0.4822, 0.4465],
    std=[0.2023, 0.1994, 0.2010]
)

train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(size=32, padding=4),
    transforms.ToTensor(),
    normalize,
])

validation_transforms = transforms.Compose([
    transforms.ToTensor(),
    normalize,
])

preprocess = transforms.Compose([
    transforms.RandomCrop(size=32, padding=4),
    transforms.ToTensor(),
    normalize,
])

def train_dataset(data_dir):
    return datasets.CIFAR10(
        root=data_dir,
        train=True,
        transform=train_transforms, 
        download=True
    )

def validation_dataset(data_dir):
    return datasets.CIFAR10(
        root=data_dir, 
        train=False, 
        transform=validation_transforms
    )

def default_epochs():
    return 200

def default_initial_lr():
    return 0.1

def default_lr_scheduler(optimizer, start_epoch=0):
    return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 120, 160, 180], last_epoch=start_epoch - 1)

def default_optimizer(model, lr, momentum, weight_decay):
    return torch.optim.SGD(model.parameters(), lr,
                            momentum=momentum,
                            weight_decay=weight_decay)

idx2label = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']