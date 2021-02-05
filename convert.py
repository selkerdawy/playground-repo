import torch

class ConvUp(torch.nn.Module):
    def __init__(self, conv, sf):
        super(ConvUp, self).__init__()
        self.conv = conv
        self.conv.stride = [x * sf for x in self.conv.stride]
        mode = 'bicubic' #'bilinear'
        self.upsample = torch.nn.Upsample(scale_factor=sf, mode=mode, align_corners=True)
    def forward(self, x):
        return self.upsample(self.conv(x))

def convert_to_conv_up(model, scale=2):
    for name, module in model._modules.items():
        if isinstance(module, (torch.nn.Conv2d)):
            model._modules[name] = ConvUp(module, scale)
        
        if len(list(module.children())) > 0:
            # recurse
            model._modules[name] = convert_to_conv_up(module, scale)
    return model

def register_forward_hook(model, function, layer_types=(torch.nn.Conv2d, torch.nn.Upsample)):
    for name, module in model._modules.items():
        if isinstance(module, layer_types):
            module.register_forward_hook(function)
        
        if len(list(module.children())) > 0:
            # recurse
            register_forward_hook(module, function)