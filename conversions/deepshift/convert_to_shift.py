import torch
import torch.nn as nn
import numpy as np
import math
import copy

import conversions.deepshift.modules_ps
import conversions.deepshift.modules_q
import conversions.deepshift.utils as utils

def convert(module, shift_type='PS', **kwargs):
    assert shift_type in ['PS', 'Q']
    if isinstance(module, torch.nn.Conv2d):
        if shift_type == 'Q':
            quant_module = conversions.deepshift.modules_q.Conv2dShiftQ(module.in_channels, module.out_channels, module.kernel_size, module.stride, module.padding, module.dilation, module.groups, module.bias is not None,
                                                                        **kwargs)
        elif shift_type == 'PS':
            quant_module = conversions.deepshift.modules_ps.Conv2dShift(module.in_channels, module.out_channels, module.kernel_size, module.stride, module.padding, module.dilation, module.groups, module.bias is not None,
                                                                       **kwargs)
    elif isinstance(module, torch.nn.Linear):
        if shift_type == 'Q':
            quant_module = converions.deepshift.modules_q.LinearShiftQ(module.in_features, module.out_features, module.bias is not None,
                                                                       **kwargs)
        elif shift_type == 'PS':
            quant_module = conversions.deepshift.modules_ps.LinearShift(module.in_features, module.out_features, module.bias is not None,
                                                                       **kwargs)

    quant_module.bias = module.bias
    quant_module.weight = module.weight
    return quant_module

def round_shift_weights(model, clone=False):
    if(clone):
        model = copy.deepcopy(model)

    for name, module in reversed(model._modules.items()):
        if len(list(module.children())) > 0:
            # recurse
            model._modules[name] = round_shift_weights(model=module)

        if type(module) == deepshift.modules.LinearShift or type(module) == deepshift.modules.Conv2dShift:
            module.shift.data = module.shift.round()
            module.sign.data = module.sign.round().sign()

            if (module.bias is not None):
                module.bias.data = utils.round_to_fixed(module.bias, fraction=16, integer=16)
        elif type(module) == deepshift.modules_q.LinearShiftQ or type(module) == deepshift.modules_q.Conv2dShiftQ:
            module.weight.data = utils.round_power_of_2(module.weight)

            if (module.bias is not None):
                module.bias.data = utils.round_to_fixed(module.bias, fraction=16, integer=16)

    return model

def count_layer_type(model, layer_type):
    count = 0
    for name, module in model._modules.items():
        if len(list(module.children())) > 0:
            # recurse
            count += count_layer_type(model=module, layer_type=layer_type)
        if type(module) == layer_type:
            count += 1

    return count