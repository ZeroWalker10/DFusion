#!/usr/bin/env python
# coding=utf-8
import sys
sys.path.append('..')

import os, json
import torch
from torch import nn
from model_zoo import ModelZoo
from dataset_zoo import DatasetZoo
from configure import *
import pdb
from attack_utils import save_one_img, load_one_img
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
from torchattacks import attacks
from torchattacks.tools.seed import random_seed
from torchattacks.tools.layer_op import FeatureAgument 
import copy

def main():
    mzoo = ModelZoo()
    dzoo = DatasetZoo()
    with open(attack_book, 'r') as fp:
        attack_targets = json.load(fp)

    device = (torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu'))
    is_targeted = True 
    is_force = False 
    for seed in [1, 1001, 2001,][::-1]:
        print('seed:', seed)
        random_seed(seed)
        for mname in ['resnet50', 'densenet121'][::-1]:
            print('model {} generates adversarial examples...'.format(mname))
    
            adv_output_dir = os.path.join(test_output_path, str(seed), mname)
            if not os.path.exists(adv_output_dir):
                os.makedirs(adv_output_dir, exist_ok=True)
    
            for (dname, dpath) in victim_datasets:
    
                adv_output_dir = os.path.join(test_output_path, str(seed), mname, dname)
                if not os.path.exists(adv_output_dir):
                    os.mkdir(adv_output_dir)
    
                print('1. dataset {} is attacked...'.format(dname)) 
                ds = dzoo.load_dataset(dname, dpath)
                label_space = list(ds.class_to_idx.values())
    
                model = mzoo.pick_model(mname)
                model = model.to(device)
                model.eval()
    
                alpha = 2 / 255.0
                max_iter = 30 
                n_samples = 20 
                for i, (attack_name, attack_args) in enumerate(baseline_attack_methods.items()):
                    adv_output_dir = os.path.join(test_output_path, str(seed), mname, dname, attack_name)
                    if not os.path.exists(adv_output_dir):
                        os.mkdir(adv_output_dir)
        
                    print('2.{} attack method {} is attacking...'.format(i, attack_name))
                    act_fusion = None
                    tmodel = copy.deepcopy(model)
                    if attack_name == 'DFusion':
                        layers = limit_layer_config[dname][mname][attack_name]
                        act_fusion = FeatureAgument(tmodel, layers)
                        attack = attacks.dfusion.DFusion(tmodel,
                                    eps=attack_args['eps'],
                                    steps=max_iter,
                                    alpha=alpha,
                                    decay=attack_args['decay_factor'],
                                    n=n_samples)
                    else:
                        # raise 'Invalid attack method!!!'
                        continue
    
                    if is_targeted:
                        attack.set_mode_targeted_by_label()
    
                    # begin to attack
                    adv_confidences = {} 
                    for (feature, label), (fname, _) in tqdm(zip(ds, ds.imgs)):
                        feature = feature.unsqueeze(0).to(device)
                        source = torch.LongTensor([label]).to(device)
    
                        fname_basename = os.path.basename(fname)
                        (_, target) = attack_targets[fname_basename]
    
                        target = torch.LongTensor([target]).to(device)
                        adv_output_file = os.path.join(adv_output_dir, fname_basename)

                        if os.path.exists(adv_output_file) and not is_force:
                            continue
    
                        adv_feature = attack(feature, target, source_labels=source, act_fusion=act_fusion)
                        save_one_img(adv_feature.detach().cpu(), adv_output_file)
    
                    if act_fusion:
                        act_fusion.clear_hook()

if __name__ == '__main__':
    main()
