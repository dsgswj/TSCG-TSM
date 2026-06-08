# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule

from mmaction.evaluation import top_k_accuracy
from mmaction.registry import MODELS
from mmaction.utils import ForwardResults, SampleList


class AvgConsensus(nn.Module):
    """Average consensus module.

    Args:
        dim (int): Decide which dim consensus function to apply.
            Defaults to 1.
    """

    def __init__(self, dim: int = 1) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Defines the computation performed at every call."""
        return x.mean(dim=self.dim, keepdim=True)


class TemporalAttentionConsensus(nn.Module):
    """Temporal Attention Consensus module.

    This module applies temporal attention mechanism to aggregate features
    across the temporal dimension. Instead of simple averaging, it learns
    to weight different temporal segments adaptively.

    Args:
        in_channels (int): Number of input channels (num_classes).
        num_segments (int): Number of temporal segments. Default: 8.
        hidden_dim (int): Hidden dimension for attention. Default: 64.
        dim (int): Decide which dim consensus function to apply.
            Defaults to 1.
    """

    def __init__(self,
                 in_channels: int,
                 num_segments: int = 8,
                 hidden_dim: int = 64,
                 dim: int = 1) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_segments = num_segments
        self.dim = dim
        self.hidden_dim = hidden_dim

        # Attention layers
        self.conv1 = nn.Conv1d(in_channels, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, 1, kernel_size=1)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()


        self.fc1 = nn.Linear(in_channels, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): Input tensor of shape (B, T, C) or (B, T, H, W, C).
                            T is the temporal dimension (num_segments).

        Returns:
            torch.Tensor: Aggregated tensor with temporal dimension reduced.
        """
        # Ensure input is 3D: (B, T, C)
        if x.dim() == 5:
            # (B, T, H, W, C) -> (B, T, C) via adaptive pooling
            B, T, H, W, C = x.shape
            x = x.permute(0, 1, 4, 2, 3)  # (B, T, C, H, W)
            x = x.mean(dim=(3, 4))  # (B, T, C)

        # x shape: (B, T, C) where T = num_segments
        B, T, C = x.shape

        x_t = x.transpose(1, 2)  # (B, C, T)

        attention = self.conv1(x_t)  # (B, hidden_dim, T)
        attention = self.tanh(attention)

        attention = self.conv2(attention)  # (B, 1, T)
        attention = attention.transpose(1, 2)  # (B, T, 1)
        attention_weights = self.sigmoid(attention)  # (B, T, 1)

        weighted_input = x * attention_weights  # (B, T, C)
        output = weighted_input.sum(dim=self.dim, keepdim=True)

        return output



class BaseHead(BaseModule, metaclass=ABCMeta):
    """Base class for head.

    All Head should subclass it.
    All subclass should overwrite:
    - :meth:`forward`, supporting to forward both for training and testing.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        loss_cls (dict): Config for building loss.
            Defaults to ``dict(type='CrossEntropyLoss', loss_weight=1.0)``.
        multi_class (bool): Determines whether it is a multi-class
            recognition task. Defaults to False.
        label_smooth_eps (float): Epsilon used in label smooth.
            Reference: arxiv.org/abs/1906.02629. Defaults to 0.
        topk (int or tuple): Top-k accuracy. Defaults to ``(1, 5)``.
        average_clips (dict, optional): Config for averaging class
            scores over multiple clips. Defaults to None.
        init_cfg (dict, optional): Config to control the initialization.
            Defaults to None.
    """

    def __init__(self,
                 num_classes: int,
                 in_channels: int,
                 loss_cls: Dict = dict(
                     type='CrossEntropyLoss', loss_weight=1.0),
                 multi_class: bool = False,
                 label_smooth_eps: float = 0.0,
                 topk: Union[int, Tuple[int]] = (1, 5),
                 average_clips: Optional[Dict] = None,
                 init_cfg: Optional[Dict] = None) -> None:
        super(BaseHead, self).__init__(init_cfg=init_cfg)
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.loss_cls = MODELS.build(loss_cls)
        self.multi_class = multi_class
        self.label_smooth_eps = label_smooth_eps
        self.average_clips = average_clips
        assert isinstance(topk, (int, tuple))
        if isinstance(topk, int):
            topk = (topk, )
        for _topk in topk:
            assert _topk > 0, 'Top-k should be larger than 0'
        self.topk = topk

    @abstractmethod
    def forward(self, x, **kwargs) -> ForwardResults:
        """Defines the computation performed at every call."""
        raise NotImplementedError

    def loss(self, feats: Union[torch.Tensor, Tuple[torch.Tensor]],
             data_samples: SampleList, **kwargs) -> Dict:
        """Perform forward propagation of head and loss calculation on the
        features of the upstream network.

        Args:
            feats (torch.Tensor | tuple[torch.Tensor]): Features from
                upstream network.
            data_samples (list[:obj:`ActionDataSample`]): The batch
                data samples.

        Returns:
            dict: A dictionary of loss components.
        """
        cls_scores = self(feats, **kwargs)
        return self.loss_by_feat(cls_scores, data_samples)

    def loss_by_feat(self, cls_scores: torch.Tensor,
                     data_samples: SampleList) -> Dict:
        """Calculate the loss based on the features extracted by the head.

        Args:
            cls_scores (torch.Tensor): Classification prediction results of
                all class, has shape (batch_size, num_classes).
            data_samples (list[:obj:`ActionDataSample`]): The batch
                data samples.

        Returns:
            dict: A dictionary of loss components.
        """
        labels = [x.gt_label for x in data_samples]
        labels = torch.stack(labels).to(cls_scores.device)
        labels = labels.squeeze()

        losses = dict()
        if labels.shape == torch.Size([]):
            labels = labels.unsqueeze(0)
        elif labels.dim() == 1 and labels.size()[0] == self.num_classes \
                and cls_scores.size()[0] == 1:
            # Fix a bug when training with soft labels and batch size is 1.
            # When using soft labels, `labels` and `cls_score` share the same
            # shape.
            labels = labels.unsqueeze(0)

        # 处理 cls_scores 形状为 (batch_size * num_segs, num_classes) 的情况
        # 而 labels 形状为 (batch_size,) 的情况
        if cls_scores.dim() == 2 and labels.dim() == 1:
            if cls_scores.size(0) % labels.size(0) == 0:
                # cls_scores 的第一维是 labels 第一维的倍数
                # 这可能是由于视频片段（segments）导致的
                num_segs = cls_scores.size(0) // labels.size(0)
                # 扩展 labels 以匹配 cls_scores
                labels = labels.repeat_interleave(num_segs)
        
        if cls_scores.size() != labels.size():
            top_k_acc = top_k_accuracy(cls_scores.detach().cpu().numpy(),
                                       labels.detach().cpu().numpy(),
                                       self.topk)
            for k, a in zip(self.topk, top_k_acc):
                losses[f'top{k}_acc'] = torch.tensor(
                    a, device=cls_scores.device)
        if self.label_smooth_eps != 0:
            if cls_scores.size() != labels.size():
                labels = F.one_hot(labels, num_classes=self.num_classes)
            labels = ((1 - self.label_smooth_eps) * labels +
                      self.label_smooth_eps / self.num_classes)

        loss_cls = self.loss_cls(cls_scores, labels)
        # loss_cls may be dictionary or single tensor
        if isinstance(loss_cls, dict):
            losses.update(loss_cls)
        else:
            losses['loss_cls'] = loss_cls
        return losses

    def predict(self, feats: Union[torch.Tensor, Tuple[torch.Tensor]],
                data_samples: SampleList, **kwargs) -> SampleList:
        """Perform forward propagation of head and predict recognition results
        on the features of the upstream network.

        Args:
            feats (torch.Tensor | tuple[torch.Tensor]): Features from
                upstream network.
            data_samples (list[:obj:`ActionDataSample`]): The batch
                data samples.

        Returns:
             list[:obj:`ActionDataSample`]: Recognition results wrapped
                by :obj:`ActionDataSample`.
        """
        cls_scores = self(feats, **kwargs)
        return self.predict_by_feat(cls_scores, data_samples)

    def predict_by_feat(self, cls_scores: torch.Tensor,
                        data_samples: SampleList) -> SampleList:
        """Transform a batch of output features extracted from the head into
        prediction results.

        Args:
            cls_scores (torch.Tensor): Classification scores, has a shape
                (B*num_segs, num_classes)
            data_samples (list[:obj:`ActionDataSample`]): The
                annotation data of every samples. It usually includes
                information such as `gt_label`.

        Returns:
            List[:obj:`ActionDataSample`]: Recognition results wrapped
                by :obj:`ActionDataSample`.
        """
        num_segs = cls_scores.shape[0] // len(data_samples)
        cls_scores = self.average_clip(cls_scores, num_segs=num_segs)
        pred_labels = cls_scores.argmax(dim=-1, keepdim=True).detach()

        for data_sample, score, pred_label in zip(data_samples, cls_scores,
                                                  pred_labels):
            data_sample.set_pred_score(score)
            data_sample.set_pred_label(pred_label)
        return data_samples

    def average_clip(self,
                     cls_scores: torch.Tensor,
                     num_segs: int = 1) -> torch.Tensor:
        """Averaging class scores over multiple clips.

        Using different averaging types ('score' or 'prob' or None,
        which defined in test_cfg) to computed the final averaged
        class score. Only called in test mode.

        Args:
            cls_scores (torch.Tensor): Class scores to be averaged.
            num_segs (int): Number of clips for each input sample.

        Returns:
            torch.Tensor: Averaged class scores.
        """

        if self.average_clips not in ['score', 'prob', None]:
            raise ValueError(f'{self.average_clips} is not supported. '
                             f'Currently supported ones are '
                             f'["score", "prob", None]')

        batch_size = cls_scores.shape[0]
        cls_scores = cls_scores.view((batch_size // num_segs, num_segs) +
                                     cls_scores.shape[1:])

        if self.average_clips is None:
            return cls_scores
        elif self.average_clips == 'prob':
            cls_scores = F.softmax(cls_scores, dim=2).mean(dim=1)
        elif self.average_clips == 'score':
            cls_scores = cls_scores.mean(dim=1)

        return cls_scores
