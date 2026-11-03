#!/usr/bin/env python
# coding=utf-8
from PIL import Image
import torch
import numpy as np
import skimage.io as io
import skimage
import pdb

def save_one_img(tensor, save_path):
    # save single picture
    if len(tensor.shape) == 4:
        # batch, channel, height, weight
        img = tensor[0].permute(1, 2, 0).detach().cpu().numpy()
        # img_tensor = tensor[0]
    else:
        img = tensor.permute(1, 2, 0).detach().cpu().numpy()
        # img_tensor = tensor
    try:
        img = Image.fromarray(np.uint8(img * 255.0)).convert('RGB')
        # img_np = torch.clamp(img_tensor * 255.0 + 0.5, 0, 255).permute(1, 2, 0).numpy().astype(np.uint8)
        # img = Image.fromarray(img_np)
        # img.save(save_path, format='PNG', compress_level=0)
        img.save(save_path)
    except Exception as e:
        pdb.set_trace()
        print('debug')

def load_one_img(img_path, resize=None):
    img = io.imread(img_path)
    if resize is not None:
        img = skimage.transform.resize(img, resize, preserve_range=True) 
    img = torch.from_numpy(img).float()
    if len(img.shape) == 2:
        img = img.unsqueeze(-1).repeat(1, 1, 3)

    try:
        img = img.permute(2, 0, 1) / 255.0
    except e:
        pdb.set_trace()
        print('debug')
    return img
