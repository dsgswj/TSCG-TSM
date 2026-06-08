# Copyright (c) OpenMMLab. All rights reserved.
"""ResNet backbone with TSM, SCGM and GRN.

This backbone combines:
1. TSM (Temporal Shift Module) for efficient temporal modeling
2. SCGM (Spatial-Channel Gated Module) for attention
3. GRN (Global Response Normalization) from ConvNeXtV2 for feature normalization

Key modification: GRN is inserted after the first convolution in the first block of each stage.

Reference:
- TSM: ICCV 2019
- SCGM: Spatial-Channel Gated Module
- GRN: ConvNeXtV2 (https://arxiv.org/abs/2301.00808)
"""

import torch
import torch.nn as nn
from mmaction.registry import MODELS
from .resnet_tsm import ResNetTSM
from .resnet import BasicBlock

# Import modules - resolve project root dynamically
import sys
import os
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
from modules.SCGM import SpatialChannelGatedModule
from modules.GRN import GRN
from mmcv.cnn.bricks import ConvModule


class BasicBlockGRN(nn.Module):
    """BasicBlock with GRN (Global Response Normalization).

    Supports two GRN usage modes:
    1. 'insert': Insert GRN after conv1→bn1→relu (preserves BN, adds GRN)
       Flow: conv1 → bn1 → relu → GRN → conv2 → bn2 → add → relu
    2. 'replace_bn': Replace BN with GRN after conv1 (removes BN, uses GRN)
       Flow: conv1 → GRN → relu → conv2 → bn2 → add → relu

    The GRN is only added in the first block of each stage where downsampling occurs.

    Uses ConvModule for TSM compatibility.

    Args:
        inplanes (int): Number of input channels.
        planes (int): Number of output channels.
        stride (int): Stride for the first convolution. Default: 1.
        downsample (nn.Module): Downsample layer for residual connection.
        use_grn (bool): Whether to use GRN. Default: False.
        grn_mode (str): GRN usage mode. 'insert' or 'replace_bn'. Default: 'insert'.
    """

    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_grn=False, grn_mode='insert'):
        super().__init__()

        self.use_grn = use_grn
        self.grn_mode = grn_mode

        # First convolution (using ConvModule for TSM compatibility)
        if use_grn and grn_mode == 'replace_bn':
            # Replace BN with GRN: conv without BN, then GRN, then ReLU
            self.conv1 = ConvModule(
                inplanes,
                planes,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
                conv_cfg=dict(type='Conv'),
                norm_cfg=None,  # No BN
                act_cfg=None  # No activation, will add manually
            )
            self.grn = GRN(planes, data_format="channels_first")
            self.relu1 = nn.ReLU(inplace=True)
        else:
            # Insert mode: conv → BN → ReLU → (optional) GRN
            self.conv1 = ConvModule(
                inplanes,
                planes,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
                conv_cfg=dict(type='Conv'),
                norm_cfg=dict(type='BN', requires_grad=True),
                act_cfg=dict(type='ReLU', inplace=True)
            )
            if use_grn and grn_mode == 'insert':
                self.grn = GRN(planes, data_format="channels_first")

        # Second convolution
        self.conv2 = ConvModule(
            planes,
            planes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            conv_cfg=dict(type='Conv'),
            norm_cfg=dict(type='BN', requires_grad=True),
            act_cfg=None  # No activation after conv2 (before residual add)
        )

        # Downsample for residual connection
        self.downsample = downsample
        self.stride = stride

        # Final activation (after residual addition)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """Forward pass with GRN."""
        identity = x

        # First conv path
        if self.use_grn and self.grn_mode == 'replace_bn':
            # replace_bn mode: conv → GRN → ReLU
            out = self.conv1(x)
            out = self.grn(out)
            out = self.relu1(out)
        else:
            # insert mode or no GRN: conv → BN → ReLU → (optional) GRN
            out = self.conv1(x)
            if self.use_grn and self.grn_mode == 'insert':
                out = self.grn(out)

        # Second conv + bn (no activation)
        out = self.conv2(out)

        # Downsample for residual connection
        if self.downsample is not None:
            identity = self.downsample(x)

        # Residual connection and final relu
        out += identity
        out = self.relu(out)

        return out


@MODELS.register_module()
class ResNetTSMSCGMGRN(ResNetTSM):
    """ResNet backbone with TSM, SCGM and GRN.

    This backbone enhances ResNetTSMSCGM by adding GRN (Global Response Normalization)
    from ConvNeXtV2.

    Supports two GRN usage modes:
    1. 'insert': Insert GRN after conv1→bn1→relu (preserves BN, adds GRN)
       - Parameters: GRN adds 2*C parameters (gamma + beta)
    2. 'replace_bn': Replace BN with GRN after conv1 (removes BN, uses GRN)
       - Parameters: GRN adds 2*C parameters, but removes 2*C BN parameters
       - Net effect: No parameter increase, but different normalization

    Supports two GRN position options:
    1. 'block': Insert GRN inside the first block of each stage (after conv1)
       - Flow: conv1 → [bn/GRN] → relu → conv2 → bn2
       - Only in stages 2, 3, 4 (where downsampling occurs)
    2. 'stage_scgm': Insert GRN between stage output and SCGM
       - Flow: stage → GRN → SCGM
       - In stages where SCGM is present

    Args:
        depth (int): Depth of ResNet (18, 34, 50, 101, 152). Default: 18.
        scgm_position (str): Where to insert SCGM.
            - 'stem': After conv1 (stem layer)
            - 'stages': After each stage output (layer1, layer2, layer3, layer4)
            - 'stages_12': After stage1 and stage2 only
            - None: Don't use SCGM
        use_grn (bool): Whether to use GRN. Default: True.
        grn_mode (str): GRN usage mode. 'insert' or 'replace_bn'. Default: 'insert'.
            Only used when grn_position='block'.
        grn_position (str): Where to insert GRN. 'block' or 'stage_scgm'. Default: 'block'.
        pretrained (str): Path to pretrained checkpoint.
        pretrained2d (bool): Whether to load 2D pretrained weights.
        num_segments (int): Number of segments for TSM. Default: 8.
        is_shift (bool): Whether to use TSM shift. Default: True.
        **kwargs: Other arguments for ResNetTSM.
    """

    def __init__(self, depth=18, scgm_position='stages_12', use_grn=True,
                 grn_mode='insert', grn_position='block',
                 pretrained=None, pretrained2d=True,
                 num_segments=8, is_shift=True, **kwargs):
        # Store configuration
        self.scgm_position = scgm_position
        self.use_grn = use_grn
        self.grn_mode = grn_mode
        self.grn_position = grn_position
        self.target_depth = depth

        # Check if ResNet18 (uses BasicBlock)
        if depth == 18:
            # ResNet18 uses BasicBlock
            from .resnet import BasicBlock as ResNetBlock
            self.resnet_block = ResNetBlock
        else:
            raise ValueError(f"Only ResNet-18 is supported, got depth={depth}")

        # Initialize parent ResNetTSM without building layers
        # We'll override _make_stage to use our custom blocks
        ResNetTSM.__init__(self, depth=depth, num_segments=num_segments, is_shift=is_shift, **kwargs)

        # Store parameters
        self.num_segments = num_segments
        self.is_shift = is_shift
        self.inplanes = 64
        self.pretrained = pretrained
        self.pretrained2d = pretrained2d

        # ResNet-18 specific parameters
        self.depth = depth
        self.base_channels = 64
        self.stages_spec = {
            18: (2, 2, 2, 2),
            34: (3, 4, 6, 3),
            50: (3, 4, 6, 3),
            101: (3, 4, 23, 3),
            152: (3, 8, 36, 3)
        }
        self.depth, self.stage_blocks = self._get_depth_settings(depth)

        # Build layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Build stages with custom blocks
        self._make_stages_with_grn()

        # Build head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Initialize TSM
        if self.is_shift:
            self.make_temporal_shift()

        # Insert SCGM and GRN (if grn_position='stage_scgm')
        if self.scgm_position is not None or self.grn_position == 'stage_scgm':
            self._insert_scgm_and_grn()

    def _get_depth_settings(self, depth):
        """Get depth settings for ResNet."""
        if depth not in self.stages_spec:
            raise ValueError(f'Invalid depth {depth}, must be one of '
                             f'{list(self.stages_spec.keys())}')
        return depth, self.stages_spec[depth]

    def _make_stages_with_grn(self):
        """Build all stages with optional GRN in first blocks."""
        channels = self.base_channels
        num_blocks_per_stage = self.stage_blocks

        for stage_idx, num_blocks in enumerate(num_blocks_per_stage):
            # Calculate output channels for this stage
            out_channels = channels * (2 ** stage_idx)

            # Build stage
            stage_blocks = []
            for block_idx in range(num_blocks):
                # Determine if this is the first block (may have downsampling)
                is_first_block = (block_idx == 0)
                stride = 2 if (stage_idx > 0 and is_first_block) else 1

                # Determine if we need downsample
                downsample = None
                if stride != 1 or self.inplanes != out_channels:
                    downsample = nn.Sequential(
                        nn.Conv2d(self.inplanes, out_channels, kernel_size=1,
                                 stride=stride, bias=False),
                        nn.BatchNorm2d(out_channels)
                    )

                # Use GRN in first block only when grn_position='block'
                use_grn = (self.use_grn and
                           is_first_block and
                           (stage_idx > 0) and
                           self.grn_position == 'block')

                # Always use BasicBlockGRN for TSM compatibility (uses ConvModule)
                # Set use_grn=False when we don't want GRN inside the block
                block = BasicBlockGRN(
                    inplanes=self.inplanes,
                    planes=out_channels,
                    stride=stride,
                    downsample=downsample,
                    use_grn=use_grn,
                    grn_mode=self.grn_mode
                )
                stage_blocks.append(block)
                self.inplanes = out_channels

            # Add stage to model
            stage = nn.Sequential(*stage_blocks)
            setattr(self, f'layer{stage_idx + 1}', stage)

    def _insert_scgm_and_grn(self):
        """Insert SCGM and GRN modules at specified positions.

        When grn_position='stage_scgm', GRN is inserted between stage output and SCGM.
        Flow: stage → GRN → SCGM
        """
        if self.scgm_position == 'stem':
            # Insert SCGM after conv1 (stem)
            self.scgm_stem = SpatialChannelGatedModule(64)

        elif self.scgm_position == 'stages':
            # Insert SCGM after each stage output
            self.scgm_layer1 = SpatialChannelGatedModule(64)
            self.scgm_layer2 = SpatialChannelGatedModule(128)
            self.scgm_layer3 = SpatialChannelGatedModule(256)
            self.scgm_layer4 = SpatialChannelGatedModule(512)

            # Insert GRN between stage and SCGM if configured
            if self.use_grn and self.grn_position == 'stage_scgm':
                self.grn_layer1 = GRN(64, data_format="channels_first")
                self.grn_layer2 = GRN(128, data_format="channels_first")
                self.grn_layer3 = GRN(256, data_format="channels_first")
                self.grn_layer4 = GRN(512, data_format="channels_first")

        elif self.scgm_position == 'stages_12':
            # Insert SCGM after stage1 and stage2 only
            self.scgm_layer1 = SpatialChannelGatedModule(64)
            self.scgm_layer2 = SpatialChannelGatedModule(128)

            # Insert GRN between stage and SCGM if configured
            if self.use_grn and self.grn_position == 'stage_scgm':
                self.grn_layer1 = GRN(64, data_format="channels_first")
                self.grn_layer2 = GRN(128, data_format="channels_first")

        # If no SCGM but GRN is requested at stage_scgm position, still create GRN
        if self.scgm_position is None and self.use_grn and self.grn_position == 'stage_scgm':
            self.grn_layer1 = GRN(64, data_format="channels_first")
            self.grn_layer2 = GRN(128, data_format="channels_first")
            self.grn_layer3 = GRN(256, data_format="channels_first")
            self.grn_layer4 = GRN(512, data_format="channels_first")

    def forward(self, x):
        """Forward with optional GRN and SCGM.

        Args:
            x (Tensor): Input tensor (N, C, H, W) or (N*T, C, H, W).

        Returns:
            Tensor: Output features from the final stage.
        """
        # Stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # SCGM after stem if configured
        if self.scgm_position == 'stem' and hasattr(self, 'scgm_stem'):
            x = self.scgm_stem(x)

        # Stage 1
        x = self.layer1(x)
        # Apply GRN if positioned at stage_scgm
        if self.use_grn and self.grn_position == 'stage_scgm' and hasattr(self, 'grn_layer1'):
            x = self.grn_layer1(x)
        # Apply SCGM
        if self.scgm_position in ['stages', 'stages_12'] and hasattr(self, 'scgm_layer1'):
            x = self.scgm_layer1(x)

        # Stage 2
        x = self.layer2(x)
        # Apply GRN if positioned at stage_scgm
        if self.use_grn and self.grn_position == 'stage_scgm' and hasattr(self, 'grn_layer2'):
            x = self.grn_layer2(x)
        # Apply SCGM
        if self.scgm_position in ['stages', 'stages_12'] and hasattr(self, 'scgm_layer2'):
            x = self.scgm_layer2(x)

        # Stage 3
        x = self.layer3(x)
        # Apply GRN if positioned at stage_scgm
        if self.use_grn and self.grn_position == 'stage_scgm' and hasattr(self, 'grn_layer3'):
            x = self.grn_layer3(x)
        # Apply SCGM
        if self.scgm_position == 'stages' and hasattr(self, 'scgm_layer3'):
            x = self.scgm_layer3(x)

        # Stage 4
        x = self.layer4(x)
        # Apply GRN if positioned at stage_scgm
        if self.use_grn and self.grn_position == 'stage_scgm' and hasattr(self, 'grn_layer4'):
            x = self.grn_layer4(x)
        # Apply SCGM
        if self.scgm_position == 'stages' and hasattr(self, 'scgm_layer4'):
            x = self.scgm_layer4(x)

        return x


@MODELS.register_module()
class ResNetTSMSCGMGRN18(ResNetTSMSCGMGRN):
    """ResNet-18 with TSM, SCGM and GRN.

    Convenience class for ResNet-18 configuration:
    - 4 stages with (2, 2, 2, 2) blocks
    - Supports two GRN positions: 'block' or 'stage_scgm'
    - SCGM after stage 1 and 2 (default)

    Args:
        scgm_position (str): SCGM position. Default: 'stages_12'.
        use_grn (bool): Whether to use GRN. Default: True.
        grn_mode (str): GRN usage mode. 'insert' or 'replace_bn'. Default: 'insert'.
        grn_position (str): GRN position. 'block' or 'stage_scgm'. Default: 'block'.
        num_segments (int): Number of TSM segments. Default: 8.
        is_shift (bool): Whether to use TSM shift. Default: True.
        pretrained (str): Path to pretrained weights.
        **kwargs: Other arguments.
    """

    def __init__(self, scgm_position='stages_12', use_grn=True,
                 grn_mode='insert', grn_position='block',
                 num_segments=8, is_shift=True,
                 pretrained=None, pretrained2d=True, **kwargs):
        super().__init__(
            depth=18,
            scgm_position=scgm_position,
            use_grn=use_grn,
            grn_mode=grn_mode,
            grn_position=grn_position,
            pretrained=pretrained,
            pretrained2d=pretrained2d,
            num_segments=num_segments,
            is_shift=is_shift,
            **kwargs
        )


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 60)
    print("Testing ResNetTSMSCGMGRN18")
    print("=" * 60)

    # Test ResNetTSMSCGMGRN18
    print("\n1. Testing ResNetTSMSCGMGRN18 (with GRN)")
    model = ResNetTSMSCGMGRN18(
        scgm_position='stages_12',
        use_grn=True,
        num_segments=8,
        is_shift=True,
        shift_div=4
    ).to(device)
    x = torch.randn(2, 3, 8, 224, 224).to(device)  # [B, C, T, H, W]
    x = x.reshape(2 * 8, 3, 224, 224)  # [B*T, C, H, W]
    with torch.no_grad():
        output = model(x)
    params = sum(p.numel() for p in model.parameters())
    print(f"Input:  {x.shape}")
    print(f"Output: {output.shape}")
    print(f"Parameters: {params:,}")

    # Test without GRN
    print("\n2. Testing ResNetTSMSCGMGRN18 (without GRN)")
    model_no_grn = ResNetTSMSCGMGRN18(
        scgm_position='stages_12',
        use_grn=False,
        num_segments=8,
        is_shift=True,
        shift_div=4
    ).to(device)
    with torch.no_grad():
        output_no_grn = model_no_grn(x)
    params_no_grn = sum(p.numel() for p in model_no_grn.parameters())
    print(f"Input:  {x.shape}")
    print(f"Output: {output_no_grn.shape}")
    print(f"Parameters: {params_no_grn:,}")

    # Compare parameters
    grn_params = params - params_no_grn
    print(f"\nGRN adds: {grn_params:,} parameters")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
