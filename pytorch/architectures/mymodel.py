#! /usr/bin/env python
import torch


class MyModel(torch.nn.Module):
    def __init__(self, backbone):
        super(MyModel, self).__init__()
        self.backbone = backbone

    def forward(self, x):
        body_feats = self.backbone(x)
        return body_feats



