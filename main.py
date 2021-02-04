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

def add_upsample(model):

    def print_mean(m, i, o):
        print(m.__class__.__name__, ' ----> Mean: ', nn.AdaptiveAvgPool2d(1) (o).squeeze().mean())

    nw_sq = []
    for layer in model.features.children():
        if isinstance(layer, nn.Conv2d):
            scale_factor = args.scale
            layer.stride = [x * scale_factor for x in layer.stride]
            layer.register_forward_hook(print_mean)
            mode = 'bicubic' #'bilinear'
            upsample = nn.Upsample(scale_factor=scale_factor, mode=mode, align_corners=True)
            nw_sq += [layer, upsample]
            upsample.register_forward_hook(print_mean)
        else:
            nw_sq += [layer]

    model.features = nn.Sequential(*nw_sq)

def main():
    image = image_loader(args.image) #Image filename

    model = models.vgg11_bn(pretrained=True)
    #print(model)
    add_upsample(model)
    #print(model)
    model.to(device)
    model.eval()
    probabilities = model(image)
    classid = probabilities.max(1)[1].item()
    label = map_classid_to_label(classid)
    print("Prediction is %s with logit %.3f" %(label, probabilities[0][classid]))

if __name__ == '__main__':
    main()
