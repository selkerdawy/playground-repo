import torch

class ConvUp(torch.nn.Module):
    def __init__(self, conv, sf, mode='bilinear'):
        super(ConvUp, self).__init__()
        self.conv = conv
        self.conv.stride = [x * sf for x in self.conv.stride]
        #mode = bicubic'
        self.upsample = torch.nn.Upsample(scale_factor=sf, mode=mode, align_corners=True)

    def forward(self, x):
        return self.upsample(self.conv(x))

def count_layer_type(model, layer_type=torch.nn.Conv2d, count=0):
    for name, module in model._modules.items():
        if isinstance(module, layer_type):
            count += 1
        
        if len(list(module.children())) > 0:
            # recurse
            count += count_layer_type(module, layer_type, 0)
    return count    

def convert_to_conv_up(model, scale=2, index_start=0, index_end=-1, index=0):
    if index_end < 0:
        index_end = index_end + count_layer_type(model, torch.nn.Conv2d)
    for name, module in model._modules.items():
        if isinstance(module, (torch.nn.Conv2d)):
            if index >= index_start and index <= index_end:
                model._modules[name] = ConvUp(module, scale)
            index += 1
        
        if len(list(module.children())) > 0:
            # recurse
            model._modules[name], index = convert_to_conv_up(module, scale, index_start, index_end, index)
    return model, index

def register_forward_hook(model, function, layer_types=(torch.nn.Conv2d, torch.nn.Upsample)):
    for name, module in model._modules.items():
        if isinstance(module, layer_types):
            module.register_forward_hook(function)
        
        if len(list(module.children())) > 0:
            # recurse
            register_forward_hook(module, function, layer_types)