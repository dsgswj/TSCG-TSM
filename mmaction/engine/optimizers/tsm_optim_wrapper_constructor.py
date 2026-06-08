# Copyright (c) OpenMMLab. All rights reserved.
import sys
import os
import torch
from mmengine.optim import DefaultOptimWrapperConstructor
from mmengine.utils.dl_utils.parrots_wrapper import (SyncBatchNorm_,
                                                     _BatchNorm, _ConvNd)

from mmaction.registry import OPTIM_WRAPPER_CONSTRUCTORS

# Ensure project root is in path for modules imports
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Import alpha-beta fusion modules for optimizer support
try:
    from mmaction.models.backbones.alpha_beta_fusion import (
        AlphaBetaFusion, AdaptiveAlphaBetaFusion, SpatialAlphaBetaFusion
    )
    ALPHA_BETA_MODULES = (AlphaBetaFusion, AdaptiveAlphaBetaFusion, SpatialAlphaBetaFusion)
except ImportError:
    ALPHA_BETA_MODULES = ()

# Import StarReLU for optimizer support
try:
    from modules.ace_enhanced import StarReLU
    STARRELU_AVAILABLE = True
except ImportError:
    STARRELU_AVAILABLE = False
    StarReLU = None

# Import ConvNeXt modules for optimizer support
try:
    from modules.GRN import LayerNorm as ConvNeXtLayerNorm, GRN
    CONVNEXT_MODULES = (ConvNeXtLayerNorm, GRN)
except ImportError:
    CONVNEXT_MODULES = ()


@OPTIM_WRAPPER_CONSTRUCTORS.register_module()
class TSMOptimWrapperConstructor(DefaultOptimWrapperConstructor):
    """Optimizer constructor in TSM model.

    This constructor builds optimizer in different ways from the default one.

    1. Parameters of the first conv layer have default lr and weight decay.
    2. Parameters of BN layers have default lr and zero weight decay.
    3. If the field "fc_lr5" in paramwise_cfg is set to True, the parameters
       of the last fc layer in cls_head have 5x lr multiplier and 10x weight
       decay multiplier.
    4. Weights of other layers have default lr and weight decay, and biases
       have a 2x lr multiplier and zero weight decay.
    """

    def add_params(self, params, model, **kwargs):
        """Add parameters and their corresponding lr and wd to the params.

        Args:
            params (list): The list to be modified, containing all parameter
                groups and their corresponding lr and wd configurations.
            model (nn.Module): The model to be trained with the optimizer.
        """
        # use fc_lr5 to determine whether to specify higher multi-factor
        # for fc layer weights and bias.
        fc_lr5 = self.paramwise_cfg['fc_lr5']
        first_conv_weight = []
        first_conv_bias = []
        normal_weight = []
        normal_bias = []
        lr5_weight = []
        lr10_bias = []
        bn = []

        conv_cnt = 0

        for m in model.modules():
            if isinstance(m, _ConvNd):
                m_params = list(m.parameters())
                conv_cnt += 1
                if conv_cnt == 1:
                    first_conv_weight.append(m_params[0])
                    if len(m_params) == 2:
                        first_conv_bias.append(m_params[1])
                else:
                    normal_weight.append(m_params[0])
                    if len(m_params) == 2:
                        normal_bias.append(m_params[1])
            elif isinstance(m, torch.nn.Linear):
                m_params = list(m.parameters())
                normal_weight.append(m_params[0])
                if len(m_params) == 2:
                    normal_bias.append(m_params[1])
            elif isinstance(m,
                            (_BatchNorm, SyncBatchNorm_, torch.nn.GroupNorm,
                             torch.nn.LayerNorm)):
                for param in list(m.parameters()):
                    if param.requires_grad:
                        bn.append(param)
            elif isinstance(m, ALPHA_BETA_MODULES):
                # Alpha-beta fusion parameters: treat as normal weights
                # These are learnable scalar parameters for fusion
                for param in list(m.parameters()):
                    if param.requires_grad:
                        normal_weight.append(param)
            elif STARRELU_AVAILABLE and isinstance(m, StarReLU):
                # StarReLU parameters (scale and bias): treat as normal weights
                # These are learnable scalar parameters for activation function
                for param in list(m.parameters()):
                    if param.requires_grad:
                        normal_weight.append(param)
            elif isinstance(m, CONVNEXT_MODULES):
                # ConvNeXt LayerNorm and GRN: treat like BN (zero weight decay)
                for param in list(m.parameters()):
                    if param.requires_grad:
                        bn.append(param)
            elif len(m._modules) == 0:
                if len(list(m.parameters())) > 0:
                    raise ValueError(f'New atomic module type: {type(m)}. '
                                     'Need to give it a learning policy')

        # pop the cls_head fc layer params
        last_fc_weight = normal_weight.pop()
        last_fc_bias = normal_bias.pop()
        if fc_lr5:
            lr5_weight.append(last_fc_weight)
            lr10_bias.append(last_fc_bias)
        else:
            normal_weight.append(last_fc_weight)
            normal_bias.append(last_fc_bias)

        params.append({
            'params': first_conv_weight,
            'lr': self.base_lr,
            'weight_decay': self.base_wd
        })
        params.append({
            'params': first_conv_bias,
            'lr': self.base_lr * 2,
            'weight_decay': 0
        })
        params.append({
            'params': normal_weight,
            'lr': self.base_lr,
            'weight_decay': self.base_wd
        })
        params.append({
            'params': normal_bias,
            'lr': self.base_lr * 2,
            'weight_decay': 0
        })
        params.append({'params': bn, 'lr': self.base_lr, 'weight_decay': 0})
        params.append({
            'params': lr5_weight,
            'lr': self.base_lr * 5,
            'weight_decay': self.base_wd
        })
        params.append({
            'params': lr10_bias,
            'lr': self.base_lr * 10,
            'weight_decay': 0
        })
