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

os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[2]
device = "cuda" if torch.cuda.is_available() else "cpu"

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

    nw_sq = []
    for layer in model.features.children():
        if isinstance(layer, nn.Conv2d):
            layer.stride = [x * 2 for x in layer.stride]
            nw_sq += [layer, nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)]
        else:
            nw_sq += [layer]

    model.features = nn.Sequential(*nw_sq)

def main():
    image = image_loader(sys.argv[1]) #Image filename

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
