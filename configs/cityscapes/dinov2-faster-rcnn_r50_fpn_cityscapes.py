_base_ = [
    '../_base_/models/dinov2_fast-rcnn_r50_fpn.py',
    '../_base_/datasets/cityscapes_detection.py',
    '../_base_/default_runtime.py',
    # '../_base_/schedules/schedule_1x.py'
]
img_scales = (1024, 1024)
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=img_scales, keep_ratio=False),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]
train_per_gpu_batch_size = 8
train_dataloader = dict(batch_size=train_per_gpu_batch_size,
                        dataset=dict(filter_cfg=dict(filter_empty_gt=False), pipeline=train_pipeline))

model = dict(
    backbone=dict(type='FrozenDINOViT'),
    roi_head=dict(
        bbox_head=dict(
            num_classes=8,
            loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))))


# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=0.0001,  # 0.0002 for DeformDETR
        weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.1)})
)  # custom_keys contains sampling_offsets and reference_points in DeformDETR  # noqa
# learning policy
max_epochs = 12
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=12)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[11],
        gamma=0.1)
]

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (2 samples per GPU)
auto_scale_lr = dict(base_batch_size=16)
