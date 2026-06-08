# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmengine.optim import DefaultOptimWrapperConstructor
from mmengine.utils.dl_utils.parrots_wrapper import (SyncBatchNorm_,
                                                     _BatchNorm, _ConvNd)

from mmaction.registry import OPTIM_WRAPPER_CONSTRUCTORS

# Import alpha-beta fusion modules for optimizer support
try:
    from mmaction.models.backbones.alpha_beta_fusion import (
        AlphaBetaFusion, AdaptiveAlphaBetaFusion, SpatialAlphaBetaFusion
    )
    ALPHA_BETA_MODULES = (AlphaBetaFusion, AdaptiveAlphaBetaFusion, SpatialAlphaBetaFusion)
except ImportError:
    ALPHA_BETA_MODULES = ()


@OPTIM_WRAPPER_CONSTRUCTORS.register_module()
class TSMAdamOptimWrapperConstructor(DefaultOptimWrapperConstructor):
    """Adam optimizer constructor for TSM model.

    This constructor builds Adam optimizer with different parameter groups,
    adapted from TSMOptimWrapperConstructor but optimized for Adam.

    1. Parameters of the first conv layer have default lr and weight decay.
    2. Parameters of BN layers have default lr and zero weight decay.
    3. If the field "fc_lr5" in paramwise_cfg is set to True, the parameters
       of the last fc layer in cls_head have 5x lr multiplier and zero weight
       decay (Adam typically uses lower weight decay).
    4. Weights of other layers have default lr and weight decay, and biases
       have a 2x lr multiplier and zero weight decay.

    Note: Adam optimizer typically uses lower weight decay values (e.g.,
    0.0001 or 0.00001) compared to SGD. Adjust base_wd accordingly in your
    config.

    Args:
        optim_wrapper_cfg (dict): The config dict of the optimizer wrapper.
        paramwise_cfg (dict): Parameter-wise configs.
            - fc_lr5 (bool): Whether to use 5x/10x lr multiplier for the
                last fc layer. Default: True.
    """

    def __init__(self, optim_wrapper_cfg, paramwise_cfg=None):
        super().__init__(optim_wrapper_cfg, paramwise_cfg)
        # Remove 'momentum' from optimizer_cfg as Adam doesn't use it
        if 'momentum' in self.optimizer_cfg:
            del self.optimizer_cfg['momentum']

    def add_params(self, params, model, **kwargs):
        """Add parameters and their corresponding lr and wd to the params.

        Args:
            params (list): The list to be modified, containing all parameter
                groups and their corresponding lr and wd configurations.
            model (nn.Module): The model to be trained with the optimizer.
        """
        # use fc_lr5 to determine whether to specify higher multi-factor
        # for fc layer weights and bias.
        fc_lr5 = self.paramwise_cfg.get('fc_lr5', True)
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
            elif len(m._modules) == 0:
                if len(list(m.parameters())) > 0:
                    raise ValueError(f'New atomic module type: {type(m)}. '
                                     'Need to give it a learning policy')

        # pop the cls_head fc layer params
        # Check if normal_weight and normal_bias are not empty
        if normal_weight:
            last_fc_weight = normal_weight.pop()
        else:
            last_fc_weight = None

        if normal_bias:
            last_fc_bias = normal_bias.pop()
        else:
            last_fc_bias = None

        if fc_lr5 and last_fc_weight is not None:
            lr5_weight.append(last_fc_weight)
            if last_fc_bias is not None:
                lr10_bias.append(last_fc_bias)
        else:
            if last_fc_weight is not None:
                normal_weight.append(last_fc_weight)
            if last_fc_bias is not None:
                normal_bias.append(last_fc_bias)

        # Add parameter groups only if they are non-empty
        if first_conv_weight:
            params.append({
                'params': first_conv_weight,
                'lr': self.base_lr,
                'weight_decay': self.base_wd
            })
        if first_conv_bias:
            params.append({
                'params': first_conv_bias,
                'lr': self.base_lr * 2,
                'weight_decay': 0
            })
        if normal_weight:
            params.append({
                'params': normal_weight,
                'lr': self.base_lr,
                'weight_decay': self.base_wd
            })
        if normal_bias:
            params.append({
                'params': normal_bias,
                'lr': self.base_lr * 2,
                'weight_decay': 0
            })
        if bn:
            params.append({'params': bn, 'lr': self.base_lr, 'weight_decay': 0})
        if lr5_weight:
            params.append({
                'params': lr5_weight,
                'lr': self.base_lr * 5,
                'weight_decay': self.base_wd
            })
        if lr10_bias:
            params.append({
                'params': lr10_bias,
                'lr': self.base_lr * 10,
                'weight_decay': 0
            })
