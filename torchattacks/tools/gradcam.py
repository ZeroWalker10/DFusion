#!/usr/bin/env python
# coding=utf-8
import torch
import torch.nn.functional as F

class GradCAM:
    def __init__(self, target_layer):
        self.gradients = dict()
        self.activations = dict()
        def backward_hook(module, grad_input, grad_output):
            self.gradients['value'] = grad_output[0]
            return None
        def forward_hook(module, input, output):
            self.activations['value'] = output
            return None

        target_layer.register_forward_hook(forward_hook)
        # target_layer.register_full_backward_hook(backward_hook)
        target_layer.register_backward_hook(backward_hook)

    def forward(self, model, input_tensor, class_idx=None, retain_graph=False):
        b, c, h, w = input_tensor.size()

        logit = model(input_tensor)
        if class_idx is None:
            score = logit[:, logit.max(1)[-1]].squeeze()
        else:
            score = logit[:, class_idx].squeeze()

        model.zero_grad()
        score.backward(retain_graph=retain_graph)
        gradients = self.gradients['value']
        activations = self.activations['value']
        b, k, u, v = gradients.size()

        alpha = gradients.view(b, k, -1).mean(2)
        weights = alpha.view(b, k, 1, 1)

        saliency_map = (weights * activations).sum(1, keepdim=True)
        saliency_map = F.relu(saliency_map)
        saliency_map = F.interpolate(saliency_map, size=(h, w), mode='bilinear', align_corners=False)
        saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
        saliency_map = (saliency_map - saliency_map_min).div(saliency_map_max - saliency_map_min).data

        return saliency_map, logit

    def __call__(self, model, input_tensor, class_idx=None, retain_graph=False):
        return self.forward(model, input_tensor, class_idx, retain_graph)
