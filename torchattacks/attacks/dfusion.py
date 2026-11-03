import torch
import torch.nn as nn
import numpy as np

import pdb
from ..attack import Attack
import kornia
from ..tools import kornia_transform 
from torch.distributions.uniform import Uniform
from scipy import stats as st
import torch.nn.functional as F
import math

class DFusion(Attack):
    def __init__(self, model, eps=8/255, alpha=2/255, steps=10, decay=1.0, n=20, res_lamb=0.2, ksize=3):
        super().__init__("DFusion", model)
        self.eps = eps
        self.steps = steps
        self.decay = decay
        self.alpha = alpha
        self.supported_mode = ["default", "targeted"]
        self.res_lamb = res_lamb
        self.ksize = ksize

        aug_p = 1.0
        self.transforms = [
            kornia_transform.RandomHorizontalFlip(same_on_batch=False, keepdim=False, p=aug_p),
            kornia_transform.RandomPerspective(0.5, "nearest", align_corners=True, same_on_batch=False, keepdim=False, p=aug_p),
            kornia_transform.RandomRotation(15.0, "nearest", align_corners=True, same_on_batch=False, keepdim=False, p=aug_p),
            kornia_transform.RandomVerticalFlip(same_on_batch=False, keepdim=False, p=0.6, p_batch=aug_p),
            kornia_transform.RandomThinPlateSpline(0.3, align_corners=True, same_on_batch=False, keepdim=False, p=aug_p),
            kornia_transform.RandomResize(0.8, p=aug_p),
            kornia_transform.RandomResize(1.2, p=aug_p),
            kornia_transform.RandomResize(1.6, p=aug_p),
            kornia_transform.RandomResize(2.0, p=aug_p),
            kornia_transform.RandomAffine((-1.0, 5.0), (0.3, 1.0), (0.4, 1.3), 0.5, resample="nearest",
                padding_mode="reflection", align_corners=True, same_on_batch=False, keepdim=False, p=aug_p,),
            kornia_transform.RandomErasing(scale=(0.01, 0.04), ratio=(0.3, 1.0), value=1, same_on_batch=False, keepdim=False, p=aug_p),
            kornia_transform.RandomElasticTransform((27, 27), (33, 31), (0.1, 1.0), align_corners=True, padding_mode="reflection",
                                                    same_on_batch=False, keepdim=False, p=aug_p),
            kornia_transform.RandomFisheye(kornia.core.tensor([-0.3, 0.3]), kornia.core.tensor([-0.3, 0.3]), kornia.core.tensor([0.9, 1.0]),
                same_on_batch=False, keepdim=False, p=aug_p),
        ]
        self.n = n

    def forward(self, images, labels, *args, **kwargs):
        r"""
        Overridden.
        """

        images = images.to(self.device)
        labels = labels.to(self.device)

        if self.targeted:
            target_labels = self.get_target_label(images, labels)
        else:
            target_labels = labels
        source_labels = kwargs.get('source_labels', None)

        loss = nn.CrossEntropyLoss()

        act_fusion = kwargs.get('act_fusion', None)

        lower = -torch.minimum(images, torch.Tensor([self.eps]).to(self.device))
        upper = torch.minimum(1.0-images, torch.Tensor([self.eps]).to(self.device))
        delta = Uniform(lower, upper).sample().to(self.device)
        repeat_images = images.repeat(self.n, 1, 1, 1).clone().detach()

        m = torch.zeros_like(delta) 
        for i in range(self.steps):
            delta.requires_grad = True
            adv_images = repeat_images + delta

            adv_images = torch.cat([
                    self.transforms[k%len(self.transforms)](adv_images[[k]])
                    for k in range(self.n)
            ])
            outputs = self.model(adv_images)
            outputs = outputs.mean(dim=0, keepdim=True)

            # Calculate loss
            if self.targeted:
                cost = -loss(outputs, target_labels)
            else:
                cost = loss(outputs, source_labels)

            # Update adversarial images
            grad = torch.autograd.grad(
                    cost, delta, retain_graph=False, create_graph=False
            )[0]
            grad = grad / torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True)
            grad = (self.res_lamb * grad + (1 - self.res_lamb) * F.avg_pool2d(grad, kernel_size=self.ksize, stride=1, padding=self.ksize//2)) + m * self.decay
            m = grad

            delta = delta + self.alpha * m 
            delta = delta.clamp(lower, upper).detach()

        adv_images = (images + delta).detach()
        return adv_images
