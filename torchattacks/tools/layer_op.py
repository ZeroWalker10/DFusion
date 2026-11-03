import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from functools import reduce
import random
import pdb
import math

def get_module_by_name(model, layer_name):
    names = layer_name.split(sep='.')
    return reduce(getattr, names, model)

class FeatureAgument:
    def __init__(self, model, layers):
        self.model = model
        self.layers = layers
        self.hook_handles = []
        self.layer_objects = {}
        self.selected = []
        self.objects = []
        
        for layer in self.layers:
            module = get_module_by_name(self.model, layer)
            handle = module.register_forward_hook(self.get_hook(layer))
            self.hook_handles.append(handle)
            self.selected.append(layer)
            self.objects.append(self.layer_objects[layer])

    
    def clear_hook(self):
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []

    def get_layers(self):
        return list(self.layers.keys())

    def get_hook(self, layer_name):
        gamma = self.layers[layer_name].get('gamma', 0.3)
        block_size = self.layers[layer_name].get('block_size', 3)
        block_ratio = self.layers[layer_name].get('block_ratio', 0.9)
        self.layer_objects[layer_name] = BlockFusion(gamma=gamma, block_size=block_size, block_ratio=block_ratio)

        def _hook(module, input, output):
            output = self.layer_objects[layer_name](output)
            return output
        return _hook

def dropblock(x, block_size=3, drop_prob=0.3):
    if len(x.shape) == 4:
        B, C, H, W = x.shape
    elif len(x.shape) == 3:
        B, H, W = x.shape
        C = 1

    gamma = drop_prob / (block_size ** 2)

    # block center
    center_mask = (torch.rand(B, C, H, W, device=x.device) < gamma).float()

    # block
    block_mask = F.max_pool2d(
        center_mask,
        kernel_size=block_size,
        stride=1,
        padding=block_size // 2
    )
    mask = 1 - block_mask

    if len(x.shape) == 4:
        return mask
    elif len(x.shape) == 3:
        return mask.squeeze(1)

class BlockFusion(nn.Module):
    def __init__(self, gamma=0.3, block_size=3, block_ratio=0.3, times=1):
        super().__init__()
        self.gamma = gamma
        self.times = times
        self.block_size = block_size
        self.block_ratio  = block_ratio

    def forward(self, x):
        new_x = x.clone()
        for _ in range(self.times):
            mask = dropblock(new_x, self.block_size, self.block_ratio)
            org_mask = mask + (1 - mask) * self.gamma
            fuse_mask = (1 - mask) * (1 - self.gamma)
            fuse_indices = np.random.choice(len(x), len(x), replace=False)
            new_x = org_mask * x + fuse_mask * x[fuse_indices]   

        return new_x 
