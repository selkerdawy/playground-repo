import argparse
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import tasks.cifar10.models as models

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

# todo
#
#def model_names():
#    ...

#
#    def preprocess:
#    ...

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

def default_lr_scheduler(optimizer, num_epochs, steps_per_epoch, start_epoch=0):
    return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 120, 160, 180], last_epoch=start_epoch - 1)

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

def default_loss_fn():
    return nn.CrossEntropyLoss()

def get_loss(output, batch, loss_fn):
    (_, target) = batch
    return torch.loss_fn(output, target)

def default_metrics():
    return topk(1,5)


def get_metrics(output, target, **kwargs):
    metrics_dict = dict()
    if "topk" in kwargs:
        acc1, acc5 = metrics.accuracy(output, target, kwargs["topk"])
        metrics_dict["acc1"] = acc1
        metrics_dict["acc5"] = acc5
    return metrics_dict

idx2label = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# todo
# 
# def idx2label:
#     ...