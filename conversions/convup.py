import torch

class ConvUp(torch.nn.Module):
    def __init__(self, conv, scale=2, mode='bilinear'):
        super(ConvUp, self).__init__()
        self.conv = conv
        self.conv.stride = [x * scale for x in self.conv.stride]
        self.upsample = torch.nn.Upsample(scale_factor=scale, mode=mode, align_corners=True)

    def forward(self, x):
        return self.upsample(self.conv(x))