"""
file - model.py
Implements the aesthemic model and emd loss used in paper.

Copyright (C) Yunxiao Shi 2017 - 2020
NIMA is released under the MIT license. See LICENSE for the fill license text.
"""

import torch
import torch.nn as nn
import torchvision

from .base_model import BaseModel


class NIMA(BaseModel):
    """Neural IMage Assessment model by Google"""
    def __init__(self, base_model, num_classes=10):
        super(NIMA, self).__init__()
        self.features = base_model.features
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.75),
            nn.Linear(in_features=25088, out_features=num_classes),
            nn.Softmax())

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out


class Encoder(BaseModel):
    def __init__(self):
        super().__init__()
        
        self.features = torchvision.models.vgg16().features

    def forward(self, inputs):
        feat = self.features(inputs)
        return feat.view(feat.size(0), -1)


class Classifier(BaseModel):
    def __init__(self, in_feat=25088, n_class=10):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.75),
            nn.Linear(in_feat, n_class),
            nn.Softmax()
        )
    
    def forward(self, inputs):
        return self.classifier(inputs)


class Discriminator(BaseModel):
    def __init__(self, h_feat=1024, in_feat=25088*2):
        super().__init__()

        self.discriminator = nn.Sequential(
            nn.Linear(in_feat, h_feat),
            nn.ReLU(),
            nn.Linear(h_feat, h_feat),
            nn.ReLU(),
            nn.Linear(h_feat, 4),
        )

    def forward(self, inputs):
        return self.discriminator(inputs)


def single_emd_loss(p, q, r=2):
    """
    Earth Mover's Distance of one sample

    Args:
        p: true distribution of shape num_classes × 1
        q: estimated distribution of shape num_classes × 1
        r: norm parameter
    """
    assert p.shape == q.shape, "Length of the two distribution must be the same"
    length = p.shape[0]
    emd_loss = 0.0
    for i in range(1, length + 1):
        emd_loss += torch.abs(sum(p[:i] - q[:i])) ** r
    return (emd_loss / length) ** (1. / r)


def emd_loss(p, q, r=2):
    """
    Earth Mover's Distance on a batch

    Args:
        p: true distribution of shape mini_batch_size × num_classes × 1
        q: estimated distribution of shape mini_batch_size × num_classes × 1
        r: norm parameters
    """
    assert p.shape == q.shape, "Shape of the two distribution batches must be the same."
    mini_batch_size = p.shape[0]
    loss_vector = []
    for i in range(mini_batch_size):
        loss_vector.append(single_emd_loss(p[i], q[i], r=r))
    return sum(loss_vector) / mini_batch_size
