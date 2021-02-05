import argparse
import sys
import os
import torch
import json
import numpy as np
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
from torch.autograd import Variable
from PIL import Image
import pdb
from contextlib import redirect_stdout

from convert import convert_to_conv_up, register_forward_hook

normalize = transforms.Normalize(
   mean=[0.485, 0.456, 0.406],
   std=[0.229, 0.224, 0.225]
)
preprocess = transforms.Compose([
   transforms.Resize(256),
   transforms.CenterCrop(224),
   transforms.ToTensor(),
   normalize
])

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='Effect of stride testing on Imagenet')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-s', '--scale', type=int, help='scale factor to multiply by stride for each conv')
parser.add_argument('-i', '--image', help='path to image')
parser.add_argument('--data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', default=True, type=lambda x:bool(distutils.util.strtobool(x)), 
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('-d', '--device', type=int, help='-1 for cpu, positive for gpu_id')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
device = "cuda" if torch.cuda.is_available() and args.device > 0 else "cpu"

def image_loader(image_name):
    """load image, returns cuda tensor"""
    image = Image.open(image_name)
    image = preprocess(image).float().unsqueeze(0) #unsqueeze to add dimension for batch size 1
    return image.to(device)

def map_classid_to_label(classid):
    class_idx = json.load(open("imagenet_class_index.json"))
    idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]
    return idx2label[classid]

def print_mean(m, i, o):
    print(m.__class__.__name__, ' ----> Mean: ', torch.mean(o), ' ---> std: ', torch.std(o))

def main():
    image = image_loader(args.image) #Image filename
    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()
    model = convert_to_conv_up(model, args.scale)
    register_forward_hook(model, print_mean)
    #print(model)
    model.to(device)
    model.eval()
    probabilities = model(image)
    classid = probabilities.max(1)[1].item()
    label = map_classid_to_label(classid)
    print("Prediction is %s with logit %.3f" %(label, probabilities[0][classid]))

if __name__ == '__main__':
    main()
