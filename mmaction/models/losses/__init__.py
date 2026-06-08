# Copyright (c) OpenMMLab. All rights reserved.
from .base import BaseWeightedLoss
from .cross_entropy_loss import CrossEntropyLoss

__all__ = ['BaseWeightedLoss', 'CrossEntropyLoss']
