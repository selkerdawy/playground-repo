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
    train_dir = os.path.join(data_dir, 'train')
    return datasets.ImageFolder(
        train_dir,
        train_transforms,
    )


def validation_dataset(data_dir):
    val_dir = os.path.join(data_dir, 'val')
    return datasets.ImageFolder(
        val_dir, 
        validation_transforms
    )

def default_epochs():
    return 90

def default_initial_lr():
    return 0.1

def default_lr_scheduler(optimizer, num_epochs, steps_per_epoch, start_epoch=0):
    return torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, last_epoch=start_epoch - 1)

def default_optimizer(model, lr, momentum, weight_decay):
    return torch.optim.SGD(model.parameters(), lr,
                            momentum=momentum,
                            weight_decay=weight_decay)

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
    return topk(1,)

def get_metrics(output, target, **kwargs):
    metrics_dict = dict()
    if "topk" in kwargs:
        acc1, acc5 = metrics.accuracy(output, target, kwargs["topk"])
        metrics_dict["acc1"] = acc1
        metrics_dict["acc5"] = acc5
    return metrics_dict

class_index_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "imagenet_class_index.json")
class_idx = json.load(open(class_index_path))
idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]