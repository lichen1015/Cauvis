# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional, Tuple, Iterable, List

import torch
from torch import Tensor, nn
from torch.nn.init import normal_

from mmdet.registry import MODELS
from .dino import DINO

def detach_everything(everything):
    if isinstance(everything, Tensor):
        return everything.detach()
    elif isinstance(everything, Iterable):
        return [detach_everything(x) for x in everything]
    else:
        return everything

@MODELS.register_module()
class FrozenDINO(DINO):

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False


    def extract_feat(self, inputs: Tensor) -> List[Tensor]:
        """Extract features from images."""
        with torch.no_grad():
            x = self.backbone(inputs)
            x = detach_everything(x)
        if self.with_neck:
            x = self.neck(x)
        return x
