# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmengine.model.weight_init import normal_init
from torch import Tensor, nn

from mmaction.registry import MODELS
from mmaction.utils import ConfigType, get_str_type
from .base import (AvgConsensus, BaseHead, TemporalAttentionConsensus)


@MODELS.register_module()
class TSMHead(BaseHead):
    """Class head for TSM.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        num_segments (int): Number of frame segments. Default: 8.
        loss_cls (dict or ConfigDict): Config for building loss.
            Default: dict(type='CrossEntropyLoss')
        spatial_type (str): Pooling type in spatial dimension. Default: 'avg'.
        consensus (dict or ConfigDict): Consensus config dict.
        dropout_ratio (float): Probability of dropout layer. Default: 0.4.
        init_std (float): Std value for Initiation. Default: 0.01.
        is_shift (bool): Indicating whether the feature is shifted.
            Default: True.
        temporal_pool (bool): Indicating whether feature is temporal pooled.
            Default: False.
        kwargs (dict, optional): Any keyword argument to be used to initialize
            the head.
    """

    def __init__(self,
                 num_classes: int,
                 in_channels: int,
                 num_segments: int = 8,
                 loss_cls: ConfigType = dict(type='CrossEntropyLoss'),
                 spatial_type: str = 'avg',
                 consensus: ConfigType = dict(type='AvgConsensus', dim=1),
                 dropout_ratio: float = 0.8,
                 init_std: float = 0.001,
                 is_shift: bool = True,
                 temporal_pool: bool = False,
                 **kwargs) -> None:
        super().__init__(num_classes, in_channels, loss_cls, **kwargs)

        self.spatial_type = spatial_type
        self.dropout_ratio = dropout_ratio
        self.num_segments = num_segments
        self.init_std = init_std
        self.is_shift = is_shift
        self.temporal_pool = temporal_pool

        consensus_ = consensus.copy()

        consensus_type = consensus_.pop('type')
        if get_str_type(consensus_type) == 'AvgConsensus':
            self.consensus = AvgConsensus(**consensus_)
            self.consensus = TemporalAttentionConsensus(**consensus_)
        elif get_str_type(consensus_type) == 'TemporalAttentionConsensus':
            # Use in_channels (feature channels), not num_classes
            # This enables temporal aggregation on feature space before classification
            consensus_['in_channels'] = in_channels
            consensus_['num_segments'] = num_segments
            self.consensus = TemporalAttentionConsensus(**consensus_)
        else:
            self.consensus = None

        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None
        self.fc_cls = nn.Linear(self.in_channels, self.num_classes)

        if self.spatial_type == 'avg':
            # use `nn.AdaptiveAvgPool2d` to adaptively match the in_channels.
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        else:
            self.avg_pool = None

    def init_weights(self) -> None:
        """Initiate the parameters from scratch."""
        normal_init(self.fc_cls, std=self.init_std)

        # Initialize TemporalAttentionConsensus if used
        if isinstance(self.consensus, (TemporalAttentionConsensus, TemporalAttentionConsensus)):
            # Initialize attention layers with Xavier initialization
            for m in [self.consensus.fc1, self.consensus.fc2]:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
            # Initialize conv layers for V2
            if isinstance(self.consensus, TemporalAttentionConsensus):
                for m in [self.consensus.conv1, self.consensus.conv2]:
                    if isinstance(m, nn.Conv1d):
                        nn.init.xavier_uniform_(m.weight)
                        if m.bias is not None:
                            nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor, num_segs: int, **kwargs) -> Tensor:
        """Defines the computation performed at every call.

        New order: Spatial pooling → Temporal aggregation → Classification

        Args:
            x (Tensor): The input data from backbone.
            num_segs (int): Number of segments (clip_len * num_clips * num_crops).

        Returns:
            Tensor: The classification scores for input samples.
        """
        # Step 1: Global average pooling (spatial)
        # Input: [N * num_segs, in_channels, H, W]
        if self.avg_pool is not None:
            x = self.avg_pool(x)
        # Output: [N * num_segs, in_channels, 1, 1]

        # Step 2: Flatten spatial dimensions
        x = torch.flatten(x, 1)
        # Output: [N * num_segs, in_channels]

        # Step 3: Reshape to separate temporal dimension
        # [N * num_segs, in_channels] → [N, num_segs, in_channels]
        x = x.view((-1, self.num_segments) + x.shape[1:])

        # Step 4: Temporal attention aggregation (on feature channels)
        # Input: [N, num_segs, in_channels]
        # Output: [N, 1, in_channels]
        if self.consensus is not None:
            x = self.consensus(x)

        # Step 5: Remove singleton temporal dimension
        # [N, 1, in_channels] → [N, in_channels]
        x = x.squeeze(1)

        # Step 6: Dropout
        if self.dropout is not None:
            x = self.dropout(x)

        # Step 7: Classification
        # [N, in_channels] → [N, num_classes]
        cls_score = self.fc_cls(x)

        return cls_score
