_base_ = [
    './base/imagenet_bs32.py',
    '../_base_/schedules/imagenet_bs1024_adamw_swin.py',
    '../_base_/default_runtime.py',
]
checkpoint = '/mnt/mydisk/medical_seg/fwwb_a007/checkpoints/convnext-small_32xb128_in1k.pth'
model = dict(
    type='ImageClassifier',
    backbone=dict(type='ConvNeXt',
                  arch='small',
                  drop_path_rate=0.4,
                  init_cfg=dict(
                      type='Pretrained',
                      checkpoint=checkpoint
                  )
                ),
    head=dict(
        type='MultiLabelLinearClsHead',
        num_classes=8,
        thr=0.5,
        in_channels=768,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0, use_sigmoid=True),
        init_cfg=None,
    ),
    init_cfg=dict(
        type='TruncNormal', layer=['Conv2d', 'Linear'], std=.02, bias=0.),
    train_cfg=dict(augments=[
        dict(type='Mixup', alpha=0.8),
        dict(type='CutMix', alpha=1.0),
    ]),
)

# dataset setting
train_dataloader = dict(batch_size=32)

# schedule setting
optim_wrapper = dict(
    optimizer=dict(lr=4e-3),
    clip_grad=None,
)

# runtime setting
custom_hooks = [dict(type='EMAHook', momentum=1e-4, priority='ABOVE_NORMAL')]

# NOTE: `auto_scale_lr` is for automatically scaling LR
# based on the actual training batch size.
# base_batch_size = (32 GPUs) x (128 samples per GPU)
auto_scale_lr = dict(base_batch_size=4096)
