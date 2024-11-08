# This config use refining bbox and `YOLOv5CopyPaste`.
# Refining bbox means refining bbox by mask while loading annotations and
# transforming after `YOLOv5RandomAffine`

deepen_factor = 1.00
widen_factor = 1.25

# ========================modified parameters======================
last_stage_out_channels = 512

mixup_prob = 0.15
copypaste_prob = 0.3

use_mask2refine = True
min_area_ratio = 0.01  # YOLOv5RandomAffine

norm_cfg = dict(type='BN', momentum=0.03, eps=0.001)  # Normalization config
strides = [8, 16, 32]

# =======================Unmodified in most cases==================

net = dict(
    type='YOLODetector',)

backbone=dict(
    type='YOLOv8CSPDarknet',
    arch='P5',
    last_stage_out_channels=last_stage_out_channels,
    deepen_factor=deepen_factor,
    widen_factor=widen_factor,
    norm_cfg=norm_cfg,
    act_cfg=dict(type='SiLU', inplace=True))

neck=dict(
    type='YOLOv8PAFPN',
    deepen_factor=deepen_factor,
    widen_factor=widen_factor,
    in_channels=[256, 512, last_stage_out_channels],
    out_channels=[256, 512, last_stage_out_channels],
    num_csp_blocks=3,
    norm_cfg=norm_cfg,
    act_cfg=dict(type='SiLU', inplace=True)),

num_points = 72
max_lanes = 4
sample_y = range(589, 230, -20)

heads = dict(type='FEHeadV2',
             num_priors=192,
             refine_layers=3,
             fc_hidden_dim=64,
             sample_points=36)

Piou_loss_weight = 2.
cls_loss_weight = 2.
xyt_loss_weight = 0.2
seg_loss_weight = 1.0
Dliou_loss_weight = 1
Driou_loss_weight = 1

work_dirs = "work_dirs/yolov8_culane/fenetv2"

test_parameters = dict(conf_threshold=0.4, nms_thres=50, nms_topk=max_lanes)

epochs = 15
batch_size = 24 

# or 0.6e-4
optimizer = dict(type='AdamW', lr=0.6e-3)  # 3e-4 for batchsize 8
total_iter = (88880 // batch_size) * epochs
scheduler = dict(type='CosineAnnealingLR', T_max=total_iter)

eval_ep = 3
save_ep = 10

img_norm = dict(mean=[103.939, 116.779, 123.68], std=[1., 1., 1.])
ori_img_w = 1640
ori_img_h = 590
img_w = 800
img_h = 320
cut_height = 270

train_process = [
    dict(
        type='GenerateLaneLine',
        transforms=[
            dict(name='Resize',
                 parameters=dict(size=dict(height=img_h, width=img_w)),
                 p=1.0),
            dict(name='HorizontalFlip', parameters=dict(p=1.0), p=0.5),
            dict(name='ChannelShuffle', parameters=dict(p=1.0), p=0.1),
            dict(name='MultiplyAndAddToBrightness',
                 parameters=dict(mul=(0.85, 1.15), add=(-10, 10)),
                 p=0.6),
            dict(name='AddToHueAndSaturation',
                 parameters=dict(value=(-10, 10)),
                 p=0.7),
            dict(name='OneOf',
                 transforms=[
                     dict(name='MotionBlur', parameters=dict(k=(3, 5))),
                     dict(name='MedianBlur', parameters=dict(k=(3, 5)))
                 ],
                 p=0.2),
            dict(name='Affine',
                 parameters=dict(translate_percent=dict(x=(-0.1, 0.1),
                                                        y=(-0.1, 0.1)),
                                 rotate=(-10, 10),
                                 scale=(0.8, 1.2)),
                 p=0.7),
            dict(name='Resize',
                 parameters=dict(size=dict(height=img_h, width=img_w)),
                 p=1.0),
        ],
    ),
    dict(type='ToTensor', keys=['img', 'lane_line', 'seg']),
]

val_process = [
    dict(type='GenerateLaneLine',
         transforms=[
             dict(name='Resize',
                  parameters=dict(size=dict(height=img_h, width=img_w)),
                  p=1.0),
         ],
         training=False),
    dict(type='ToTensor', keys=['img']),
]

dataset_path = './data/culane'
dataset_type = 'CULane'
dataset = dict(train=dict(
    type=dataset_type,
    data_root=dataset_path,
    split='train',
    processes=train_process,
),
val=dict(
    type=dataset_type,
    data_root=dataset_path,
    split='test',
    processes=val_process,
),
test=dict(
    type=dataset_type,
    data_root=dataset_path,
    split='test',
    processes=val_process,
))

workers = 10
log_interval = 500
seed = 0
num_classes = 4 + 1
ignore_label = 255
bg_weight = 0.4
lr_update_by_epoch = False
