#!/usr/bin/env python
# coding=utf-8
import sys
sys.path.append('..')

from torchvision.models import resnet50, vgg19, inception_v3, densenet121, wide_resnet50_2, resnet18, resnet34, vgg16, resnet152
import torchvision.models as models
from torchvision import transforms
import timm
import copy
import torch.nn as nn
import torch
import pdb
# import DCL.model as dcl_model


class ModelZoo:
    def __init__(self):
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def pick_model(self, model_name, dataset='imagenet', output_silence=False, with_normal=True):
        if dataset in ['imagenet', 'caltech101',]:
            mean = self.mean
            std = self.std

        if model_name == 'resnet50':
            if dataset == 'imagenet':
                res50 = resnet50(weights='ResNet50_Weights.IMAGENET1K_V1')
            elif dataset == 'caltech101':
                res50 = resnet50()
                res50.fc = nn.Linear(2048, 101)
                state = torch.load('res50-caltech101.pth', weights_only=False)
                res50.load_state_dict(state['model_state_dict'])
            else:
                raise 'Invalid dataset!!!'

            if output_silence:
                res50.fc = nn.Identity()

            res50.eval()
            if with_normal:
                return torch.nn.Sequential(
                    transforms.Normalize(mean=mean, std=std),
                    res50
                )
            else:
                return res50
        if model_name == 'resnet152':
            if dataset == 'imagenet':
                res152 = resnet152(weights='ResNet152_Weights.IMAGENET1K_V1')
            else:
                raise 'Invalid dataset!!!'

            if output_silence:
                res152.fc = nn.Identity()

            res152.eval()
            if with_normal:
                return torch.nn.Sequential(
                    transforms.Normalize(mean=mean, std=std),
                    res152
                )
            else:
                return res152
        elif model_name == 'resnet34':
            if dataset == 'imagenet':
                res34 = resnet34(weights='ResNet34_Weights.IMAGENET1K_V1')
            else:
                raise 'Invalid dataset!!!'

            if output_silence:
                res34.fc = nn.Identity()

            res34.eval()
            if with_normal:
                return torch.nn.Sequential(
                    transforms.Normalize(mean=mean, std=std),
                    res34
                )
            else:
                return res34
        elif model_name == 'resnet18':
            if dataset == 'imagenet':
                res18 = resnet18(weights='ResNet18_Weights.IMAGENET1K_V1')
            else:
                raise 'Invalid dataset!!!'

            if output_silence:
                res18.fc = nn.Identity()

            res18.eval()
            if with_normal:
                return torch.nn.Sequential(
                    transforms.Normalize(mean=mean, std=std),
                    res18
                )
            else:
                return res18
        elif model_name == 'vgg19':
            if dataset == 'imagenet':
                v19 = vgg19(weights='VGG19_Weights.IMAGENET1K_V1')
            elif dataset == 'caltech101':
                v19 = vgg19()
                v19.classifier[6] = nn.Linear(4096, 101)
                state = torch.load('v19-caltech101.pth', weights_only=False)
                v19.load_state_dict(state['model_state_dict'])
            else:
                raise 'Invalid dataset!!!'

            if output_silence:
                v19.classifier[6] = nn.Identity()

            v19.eval()
            if with_normal:
                return torch.nn.Sequential(
                    transforms.Normalize(mean=mean, std=std),
                    v19
                )
            else:
                return v19
        elif model_name == 'vgg16':
            if dataset == 'imagenet':
                v16 = vgg16(weights='VGG16_Weights.IMAGENET1K_V1')
            else:
                raise 'Invalid dataset!!!'

            if output_silence:
                v16.classifier[6] = nn.Identity()

            v16.eval()
            if with_normal:
                return torch.nn.Sequential(
                    transforms.Normalize(mean=mean, std=std),
                    v16
                )
            else:
                return v16
        elif model_name == 'inception_v3':
            if dataset == 'imagenet':
                incep3 = inception_v3(weights='Inception_V3_Weights.IMAGENET1K_V1')
            elif dataset == 'caltech101':
                incep3 = torch.load('incep3-caltech101.pth', weights_only=False)
            else:
                raise 'Invalid dataset!!!'

            if output_silence:
                incep3.fc = nn.Identity()

            incep3.eval()
            if with_normal:
                return torch.nn.Sequential(
                    transforms.Normalize(mean=mean, std=std),
                    incep3
                )
            else:
                return incep3
        elif model_name == 'densenet121':
            if dataset == 'imagenet':
                dense121 = densenet121(weights='DenseNet121_Weights.IMAGENET1K_V1')
            elif dataset == 'caltech101':
                dense121 = densenet121()
                dense121.classifier = nn.Linear(1024, 101)
                state = torch.load('dense121-caltech101.pth', weights_only=False)
                dense121.load_state_dict(state['model_state_dict'])
            else:
                raise 'Invalid dataset!!!'

            if output_silence:
                dense121.classifier = nn.Identity() 

            dense121.eval()
            if with_normal:
                return torch.nn.Sequential(
                    transforms.Normalize(mean=mean, std=std),
                    dense121
                )
            else:
                return dense121
        elif model_name == 'wide_resnet50_2':
            if dataset == 'imagenet':
                wres50 = wide_resnet50_2(weights='Wide_ResNet50_2_Weights.IMAGENET1K_V1')
            elif dataset == 'caltech101':
                wres50 = wide_resnet50_2()
                wres50.fc = nn.Linear(2048, 101)
                state = torch.load('wres50-caltech101.pth', weights_only=False)
                wres50.load_state_dict(state['model_state_dict'])
            else:
                raise 'Invalid dataset!!!'

            if output_silence:
                wres50.fc = nn.Identity()

            wres50.eval()
            if with_normal:
                return torch.nn.Sequential(
                    transforms.Normalize(mean=mean, std=std),
                    wres50
                )
            else:
                return wres50
        elif model_name == 'swin_t':
            swin = models.swin_t(weights='IMAGENET1K_V1')
            swin.eval()
            return torch.nn.Sequential(
                transforms.Normalize(mean=self.mean, std=self.std),
                swin
            )
        elif model_name == 'swin_v2_t':
            swin = models.swin_v2_t(weights='IMAGENET1K_V1')
            swin.eval()
            return torch.nn.Sequential(
                transforms.Normalize(mean=self.mean, std=self.std),
                swin
            )
        elif model_name == 'vit_b_16':
            vit = models.vit_b_16(weights='IMAGENET1K_V1')
            vit.eval()
            return torch.nn.Sequential(
                transforms.Resize([224, 224], antialias=True),
                transforms.Normalize(mean=self.mean, std=self.std),
                vit
            )
        elif model_name == 'maxvit_t':
            vit = models.maxvit_t(weights='IMAGENET1K_V1')
            vit.eval()
            return torch.nn.Sequential(
                transforms.Resize([224, 224], antialias=True),
                transforms.Normalize(mean=self.mean, std=self.std),
                vit
            )
        elif model_name == 'inception_resnet_v2':
            incres_v2 = timm.create_model('inception_resnet_v2', pretrained=True)
            incres_v2.eval()
            if with_normal:
                return torch.nn.Sequential(
                    transforms.Normalize(mean=mean, std=std),
                    incres_v2 
                )
            else:
                return incres_v2
        else:
            raise 'Invalid model name!!!'


if __name__ == '__main__':
    mz = ModelZoo()
    model_names = ['vgg19']
    for mn in model_names:
        print(mn)
        model = mz.pick_model(mn)
        print('debug')
