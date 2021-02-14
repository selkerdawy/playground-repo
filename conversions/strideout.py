import torch
import numpy as np

class StrideOut(torch.nn.Module):
    def __init__(self, conv, scale=2, rate=0.1):
        super(StrideOut, self).__init__()
        self.conv = conv
        self.original_stride = conv.stride
        self.drop_stride = [x * scale for x in self.conv.stride]
        self.rate = rate

    def forward(self, x):
        if self.training and np.random.rand(1) < self.rate:
            self.conv.stride = self.drop_stride
        else:
            self.conv.stride = self.original_stride
        return self.conv(x)