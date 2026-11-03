#!/usr/bin/env python
# coding=utf-8
import sys
sys.path.append('..')

import os, json
import torch
from model_zoo import ModelZoo
from dataset_zoo import DatasetZoo
from configure import *
import pdb
from attack_utils import load_one_img
from attack_metric import AttackMetric
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np

with open(attack_book, 'r') as fp:
    attack_targets = json.load(fp)

device = torch.device('cuda:0')

def evaluate(evaluation_file, src_model_name, target_model_name, attakc_method, dataset_name, dataset_path, seed):
    mzoo = ModelZoo() 

    # load target model
    smodel = mzoo.pick_model(src_model_name, dataset=dataset_name)
    model = mzoo.pick_model(target_model_name, dataset=dataset_name)
    model = model.to(device)
    smodel = smodel.to(device)
    model.eval()
    smodel.eval()

    # prepare source adversarial examples
    adv_output_dir = os.path.join(test_output_path, str(seed), src_model_name, dataset_name, attack_method)

    # begin to evaluate
    adv_imgs = [fname for fname in os.listdir(adv_output_dir) if not fname.endswith('.npy')]
    src_labels, target_labels, white_confidences, black_confidences = [], [], [], []
    k = 0
    for fname in tqdm(adv_imgs):
        fpath = os.path.join(adv_output_dir, fname)
        feature = load_one_img(fpath)
        feature = feature.unsqueeze(0).to(device)

        try:
            (src, target) = attack_targets[fname]
        except Exception as e:
            continue

        src_labels.append(src)
        target_labels.append(target)

        with torch.no_grad():
            adv_logits = model(feature)
            adv_confidence = F.softmax(adv_logits, dim=1)
            white_confidence = F.softmax(smodel(feature), dim=1)
        white_confidences.append(white_confidence.detach().cpu().numpy())
        black_confidences.append(adv_confidence.detach().cpu().numpy())

    black_confidences = np.concatenate(black_confidences)
    white_confidences = np.concatenate(white_confidences)
    src_labels = np.array(src_labels)
    target_labels = np.array(target_labels)

    am = AttackMetric(src_labels, target_labels, white_confidences, black_confidences)
    prefix = '{},{},{},{:.4f},{:.4f},'.format(attack_method, src_model_name,
                                    target_model_name, am.error_rate(),
                                    am.targeted_success_rate())
    suffix = ''
    line = prefix + suffix
    print(line)
    with open(evaluation_file, 'a') as fp:
        fp.write(line + '\n')

if __name__ == '__main__':
    # for src_mname in ['resnet50']:
    for src_mname in ['densenet121']:
        for target_mname in victim_model_names:
            for ds_name, ds_path in victim_datasets:
                for attack_method in baseline_attack_methods.keys():
                    # for seed in [1, 1001, 2001]:
                    # for seed in [1]:
                    for seed in [2001]:
                        evaluate(mul_test_evaluation_file, src_mname, target_mname, attack_method, ds_name, ds_path, seed)
    print('===evaluate end===')

