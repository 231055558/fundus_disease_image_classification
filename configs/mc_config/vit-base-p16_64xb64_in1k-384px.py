_base_ = [
    '../_base_/models/vit-base-p16.py',
    './base/imagenet_bs32.py',
    '../_base_/schedules/imagenet_bs4096_AdamW.py',
    '../_base_/default_runtime.py'
]
model = dict(
    backbone=dict(
        img_size=384,
        init_cfg=[
            dict(
                type='Pretrained',
                # layer='Conv2d',
                # mode='fan_in',
                # nonlinearity='linear',
                checkpoint='/mnt/mydisk/medical_seg/fwwb_a007/checkpoints/vit-base-p16_in21k-pre-3rdparty_ft-64xb64_in1k.pth')
        ]
    ),
    head=dict(
        type='VisionTransformerClsHead',
        num_classes=8,
        loss=dict(
            type='LabelSmoothLoss', label_smooth_val=0.01,
            mode='multi_label'),
    ))
# model setting
# model = dict(backbone=dict(img_size=384))

# dataset setting
data_preprocessor = dict(
    mean=[127.5, 127.5, 127.5],
    std=[127.5, 127.5, 127.5],
    # convert image from BGR to RGB
    to_rgb=True,
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', scale=384, backend='pillow'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='ResizeEdge', scale=384, edge='short', backend='pillow'),
    dict(type='CenterCrop', crop_size=384),
    dict(type='PackInputs'),
]

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = dict(dataset=dict(pipeline=test_pipeline))

# schedule setting
optim_wrapper = dict(clip_grad=dict(max_norm=1.0))
