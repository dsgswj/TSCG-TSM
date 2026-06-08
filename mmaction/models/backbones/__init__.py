# Copyright (c) OpenMMLab. All rights reserved.
from .resnet import ResNet
from .resnet_tsm import ResNetTSM
from .TSCG_TSM import ResNetTSMSCGMGRN, ResNetTSMSCGMGRN18

__all__ = [
    'ResNet', 'ResNetTSM',
    'ResNetTSMSCGMGRN', 'ResNetTSMSCGMGRN18',
]
