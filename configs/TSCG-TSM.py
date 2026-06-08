# dataset settings
dataset_type = 'VideoDataset'
data_root = 'tools/data/'

ann_file_train = 'tools/data/trainlist.txt'
ann_file_val = 'tools/data/testlist.txt'

file_client_args = dict(io_backend='disk')

num_segments = 8
shift_div = 4
milestones = [30, 60, 80]

# Model configuration
model = dict(
    type='Recognizer2D',
    backbone=dict(
        type='ResNetTSMSCGMGRN18',
        pretrained='torchvision://resnet18',
        pretrained2d=True,
        norm_eval=False,
        num_segments=num_segments,
        is_shift=True,
        shift_div=shift_div,
        scgm_position='stages',
        use_grn=True,
        grn_position='stage_scgm',
        grn_mode='insert',
    ),
    cls_head=dict(
        type='TSMHead',
        in_channels=512, 
        num_classes=4,
        num_segments=num_segments,
        spatial_type='avg',
        consensus=dict(
            type='TemporalAttentionConsensus',
            dim=1
        ),
        dropout_ratio=0.7,
        init_std=0.001,
        is_shift=True,
        average_clips='prob'
    ),
    data_preprocessor=dict(
        type='ActionDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375]
    )
)

# Training pipeline
train_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=16, use_random_offset=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(
        type='MultiScaleCrop',
        input_size=224,
        scales=(1, 0.875, 0.75, 0.66),
        random_crop=False,
        max_wh_scale_gap=1,
        num_fixed_crops=13),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='PackActionInputs')
]

# Validation pipeline
val_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(
        type='SampleFrames',
        clip_len=1,
        frame_interval=1,
        num_clips=16,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='PackActionInputs')
]

# Test pipeline
test_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(
        type='SampleFrames',
        clip_len=1,
        frame_interval=1,
        num_clips=16,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='PackActionInputs')
]

# Data loaders
train_dataloader = dict(
    batch_size=16,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=dict(video=data_root),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=16,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=dict(video=data_root),
        pipeline=val_pipeline,
        test_mode=True))

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=dict(video=data_root),
        pipeline=test_pipeline,
        test_mode=True))

val_evaluator = dict(type='AccMetric')
test_evaluator = val_evaluator

# Optimizer
optim_wrapper = dict(
    constructor='TSMOptimWrapperConstructor',
    paramwise_cfg=dict(fc_lr5=True),
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.00001),
    clip_grad=dict(max_norm=20, norm_type=2))

# Learning rate scheduler
param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=100,
        by_epoch=True,
        milestones=milestones,
        gamma=0.1)
]

# Training configuration
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=100, val_begin=1, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# Default hooks
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=3, max_keep_ckpts=3, save_best='auto'),
    logger=dict(type='LoggerHook', interval=1000, ignore_last=False),
    timer=dict(type='IterTimerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    runtime_info=dict(type='RuntimeInfoHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    sync_buffers=dict(type='SyncBuffersHook')
)

# Custom hooks
custom_hooks = [
    dict(
        type='ValidationMonitorHook',
        save_to_file=True
    )
]

# Log processor
log_processor = dict(
    type='LogProcessor',
    window_size=1,
    by_epoch=True,
)

# Runtime settings
default_scope = 'mmaction'


# Log level
log_level = 'INFO'

# Load from
load_from = None
resume = False

# Randomness
randomness = dict(
    seed=42,
    diff_rank_seed=False,
    deterministic=False
)

# Auto scale LR
auto_scale_lr = dict(enable=False, base_batch_size=128)

# Visualization
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='ActionVisualizer',
    vis_backends=vis_backends,
    save_dir='work_dirs/TSCG-TSM')
