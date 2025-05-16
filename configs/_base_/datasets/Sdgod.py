# dataset settings
dataset_type = 'SdgodDataset'
data_root = '/mnt/datasets/Single-DGOD/'

# Example to use different file client
# Method 1: simply set the data root and let the file I/O module
# automatically infer from prefix (not support LMDB and Memcache yet)

# data_root = 's3://openmmlab/datasets/detection/coco/'

# Method 2: Use `backend_args`, `file_client_args` in versions before 3.0.0rc6
# backend_args = dict(
#     backend='petrel',
#     path_mapping=dict({
#         './data/': 's3://openmmlab/datasets/detection/',
#         'data/': 's3://openmmlab/datasets/detection/'
#     }))
backend_args = None

img_scales = (640, 640)
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=img_scales, keep_ratio=False),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=img_scales, keep_ratio=False),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
train_dataloader = dict(
    batch_size=2,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='Daytime_Sunny/daytime_clear/VOC2007/ImageSets/Main/train.txt',
        data_prefix=dict(sub_data_root='Daytime_Sunny/daytime_clear/VOC2007/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        backend_args=backend_args))
val_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type="ConcatDataset",
        ignore_keys=['dataset_type'],
        datasets=[
            dict(type=dataset_type, data_root=data_root,  # Daytime_Sunny
                 ann_file='Daytime_Sunny/daytime_clear/VOC2007/ImageSets/Main/test.txt',
                 data_prefix=dict(sub_data_root='Daytime_Sunny/daytime_clear/VOC2007/'),
                 filter_cfg=dict(
                     filter_empty_gt=True, min_size=32, bbox_min_size=32),
                 pipeline=test_pipeline),
            dict(type=dataset_type, data_root=data_root,  # Daytime-Foggy
                 ann_file='Daytime-Foggy/daytime_foggy/VOC2007/ImageSets/Main/train.txt',
                 data_prefix=dict(sub_data_root='Daytime-Foggy/daytime_foggy/VOC2007/'),
                 filter_cfg=dict(
                     filter_empty_gt=True, min_size=32, bbox_min_size=32),
                 pipeline=test_pipeline),
            dict(type=dataset_type, data_root=data_root,  # Dusk-rainy
                 ann_file='Dusk-rainy/dusk_rainy/VOC2007/ImageSets/Main/train.txt',
                 data_prefix=dict(sub_data_root='Dusk-rainy/dusk_rainy/VOC2007/'),
                 filter_cfg=dict(
                     filter_empty_gt=True, min_size=32, bbox_min_size=32),
                 pipeline=test_pipeline),
            dict(type=dataset_type, data_root=data_root,  # Night_rainy
                 ann_file='Night_rainy/night_rainy/VOC2007/ImageSets/Main/train.txt',
                 data_prefix=dict(sub_data_root='Night_rainy/night_rainy/VOC2007/'),
                 filter_cfg=dict(
                     filter_empty_gt=True, min_size=32, bbox_min_size=32),
                 pipeline=test_pipeline),
            dict(type=dataset_type, data_root=data_root,  # Night-Sunny
                 ann_file='Night-Sunny/Night-Sunny/VOC2007/ImageSets/Main/train.txt',
                 data_prefix=dict(sub_data_root='Night-Sunny/Night-Sunny/VOC2007/'),
                 filter_cfg=dict(
                     filter_empty_gt=True, min_size=32, bbox_min_size=32),
                 pipeline=test_pipeline),
        ])
)
test_dataloader = val_dataloader

val_evaluator = dict(type='SDGODMetric',
                     dataset_keys=['Daytime_Sunny', 'Daytime-Foggy', 'Dusk-rainy', 'Night_rainy', 'Night-Sunny'],
                     eval_mode='11points', metric=['mAP'], )
test_evaluator = val_evaluator

# inference on test dataset and
# format the output results for submission.
# test_dataloader = dict(
#     batch_size=1,
#     num_workers=2,
#     persistent_workers=True,
#     drop_last=False,
#     sampler=dict(type='DefaultSampler', shuffle=False),
#     dataset=dict(
#         type=dataset_type,
#         data_root=data_root,
#         ann_file=data_root + 'annotations/image_info_test-dev2017.json',
#         data_prefix=dict(img='test2017/'),
#         test_mode=True,
#         pipeline=test_pipeline))
# test_evaluator = dict(
#     type='CocoMetric',
#     metric='bbox',
#     format_only=True,
#     ann_file=data_root + 'annotations/image_info_test-dev2017.json',
#     outfile_prefix='./work_dirs/coco_detection/test')
