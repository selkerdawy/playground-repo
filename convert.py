import torch

def count_layer_type(model, layer_type=torch.nn.Conv2d, count=0):
    for name, module in model._modules.items():
        if isinstance(module, layer_type):
            count += 1
        
        if len(list(module.children())) > 0:
            # recurse
            count += count_layer_type(module, layer_type, 0)
    return count    

def convert(model, layer_from, layer_to, index_start=0, index_end=-1, index=0, **kwargs):
    if index_end < 0:
        index_end = index_end + count_layer_type(model, layer_from)
    for name, module in model._modules.items():
        if isinstance(module, (layer_from)):
            if index >= index_start and index <= index_end:
                module.name = name
                model._modules[name] = layer_to(module, **kwargs)
            index += 1
        
        if len(list(module.children())) > 0:
            # recurse
            model._modules[name], index = convert(module, layer_from, layer_to, index_start, index_end, index, **kwargs)
    return model, index

def register_forward_hook(model, function, layer_types=(torch.nn.Conv2d, torch.nn.Upsample)):
    for name, module in model._modules.items():
        if isinstance(module, layer_types):
            module.register_forward_hook(function)
        
        if len(list(module.children())) > 0:
            # recurse
            register_forward_hook(module, function, layer_types)