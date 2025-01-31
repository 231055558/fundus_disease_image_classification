_base_ = [
    '../_base_/models/resnet50.py', './base/imagenet_bs32.py',
    '../_base_/schedules/imagenet_bs256.py', '../_base_/default_runtime.py'
]
model = dict(backbone=dict(init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')), head=dict(type='MultiLabelLinearClsHead', thr=0.5, num_classes=8, loss=dict(use_sigmoid=True)))