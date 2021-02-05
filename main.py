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


parser = argparse.ArgumentParser(description='Effect of stride testing')
parser.add_argument('-i', '--image', help='path to image')
parser.add_argument('-a', '--arch', help='Architecture: vgg11, resnet18')
parser.add_argument('-d', '--device', type=int, help='-1 for cpu, positive for gpu_id')
parser.add_argument('-s', '--scale', type=int, help='scale factor to multiply by stride for each conv')

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
    if args.arch == 'vgg11':
        model = models.vgg11_bn(pretrained=True)
    elif args.arch == 'resnet18':
        model = models.resnet18(pretrained=True)
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
