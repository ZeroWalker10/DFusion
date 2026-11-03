import torch.nn.functional as F
from torch import nn
import torch
import numpy as np

def pick_exclusive_labels(target, n_class):
    label_set = np.arange(n_class)
    exc_set = np.setdiff1d(label_set, [target])
    return np.random.choice(exc_set)

def smooth_label(labels, alpha, n_class):
    onehot_labels = F.one_hot(labels, n_class)
    reg_labels = torch.ones_like(onehot_labels) / n_class
    new_labels = (1 - alpha) * onehot_labels + alpha * reg_labels
    return new_labels

class SmoothCrossEntropyLoss(nn.Module):
    def __init__(self, reduction='none'):
        super().__init__()

        self.reduction = reduction

    def forward(self, preds, smooth_labels):
        log_probs = F.log_softmax(preds, dim=1)
        loss = (-log_probs * smooth_labels).sum(dim=1)

        if self.reduction in ['mean', 'none']:
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            raise NotImplemented
