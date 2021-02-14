import argparse
import json
import os

import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

normalize = transforms.Normalize(
   mean=[0.485, 0.456, 0.406],
   std=[0.229, 0.224, 0.225]
)

train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
])

validation_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize,
])

preprocess = validation_transforms

def train_dataset(data_dir):
    train_dir = os.path.join(args.data, 'train')
    return datasets.ImageFolder(
        train_dir,
        train_transforms,
    )

def validation_dataset(data_dir):
    val_dir = os.path.join(args.data, 'val')
    return datasets.ImageFolder(
        val_dir, 
        validation_transforms
    )

def default_initial_lr():
    return 0.1

def default_lr_scheduler(optimizer, start_epoch):
    return torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, last_epoch=start_epoch - 1)

def default_optimizer(model, lr, momentum, weight_decay):
    return torch.optim.SGD(model.parameters(), lr,
                            momentum=momentum,
                            weight_decay=weight_decay)

class_idx = json.load(open("imagenet_class_index.json"))
idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]