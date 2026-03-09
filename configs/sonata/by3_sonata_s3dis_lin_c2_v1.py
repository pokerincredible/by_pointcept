_base_ = ["../_base_/default_runtime.py"]

# misc custom setting
batch_size = 36  # bs: total bs in all gpus
num_worker = 36
mix_prob = 0.8
clip_grad = 3.5
empty_cache = True
enable_amp = True
enable_ocnn = False
empty_cache_per_epoch = False
find_unused_parameters = False
epoch = 3000
eval_epoch = 100
sync_bn = False
amp_dtype = 'float16'

# scheduler settings
optimizer = dict(type="AdamW", lr=0.002, weight_decay=0.02)
scheduler = dict(
    type="OneCycleLR",
    max_lr=[0.0025, 0.00025],
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)
param_dicts = [dict(keyword="block", lr=0.0002)]
# model settings
model = dict(
    type="DefaultSegmentorV2",
    num_classes=2,
    backbone_out_channels=1232,
    backbone=dict(
        type="PT-v3m2",
        in_channels=6,
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(3, 3, 3, 12, 3),
        enc_channels=(48, 96, 192, 384, 512),
        enc_num_head=(3, 6, 12, 24, 32),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.25,
        shuffle_orders=True,
        pre_norm=True,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=False,
        upcast_softmax=False,
        traceable=False,
        mask_token=False,
        enc_mode=True,
        freeze_encoder=False,
    ),
    criteria=[
        dict(
            type='CrossEntropyLoss',
            loss_weight=0.6,
            ignore_index=-1,
            weight=[1.0, 3.5]
        ),
        dict(
            type='LovaszLoss',
            mode='multiclass',
            loss_weight=1.0,
            ignore_index=-1
        ),
        # dict(type='FocalLoss', alpha=0.75, gamma=2.0, loss_weight=1.0, ignore_index=-1),
        # dict(type='LovaszLoss', mode='multiclass', loss_weight=1.5, ignore_index=-1)
    ],
    freeze_backbone=False,
)



# dataset settings
dataset_type = "S3DISDataset"
data_root = "/root/autodl-fs/by3_class2_v4/train"

data = dict(
    num_classes=2,
    ignore_index=-1,
    names=['class_0', 'class_1'],
    train=dict(
        type=dataset_type,
        split=("Area_1",),
        data_root=data_root,
        transform=[
            dict(type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=1.0),
            # dict(type="RandomRotateTargetAngle", angle=(1/2, 1, 3/2), center=[0, 0, 0], axis="z", p=0.75),
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
            dict(type="RandomRotate", angle=[-0.015625, 0.015625], axis="x", p=0.5),
            dict(type="RandomRotate", angle=[-0.015625, 0.015625], axis="y", p=0.5),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            dict(type="RandomFlip", p=0.5),
            dict(type="RandomJitter", sigma=0.01, clip=0.03),
            dict(type="ElasticDistortion", distortion_params=[[0.2, 0.04], [0.8, 0.08]]),
            dict(type="ChromaticAutoContrast", p=0.2, blend_factor=None),
            dict(type="ChromaticTranslation", p=0.95, ratio=0.05),
            dict(type="ChromaticJitter", p=0.95, std=0.05),
            # dict(type="HueSaturationTranslation", hue_max=0.2, saturation_max=0.2),
            # dict(type="RandomColorDrop", p=0.2, color_augment=0.0),
            dict(
                type="GridSample",
                grid_size=0.015,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
            ),
            dict(type="SphereCrop", sample_rate=0.65, mode="random"),
            dict(type="SphereCrop", point_max=90000, mode="random"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment"),
                feat_keys=("coord", "normal"),
            ),
        ],
        test_mode=False,
        loop=30,
    ),
    val=dict(
        type=dataset_type,
        split="Area_5",
        data_root=data_root,
        transform=[

            dict(
                type="GridSample",
                grid_size=0.015,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
            ),

            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment"),
                feat_keys=("coord", "normal"),
            ),
        ],
        test_mode=False,
    ),
    test=dict(
        type=dataset_type,
        split="Area_5",
        data_root=data_root,
        transform=[
        ],
        test_mode=True,
        test_cfg=dict(
            voxelize=dict(
                type="GridSample",
                grid_size=0.015,
                hash_type="fnv",
                mode="test",
                return_grid_coord=True,
            ),
            crop=None,
            post_transform=[
                #dict(type="CenterShift", apply_z=False),
                dict(type="ToTensor"),
                dict(
                    type="Collect",
                    keys=("coord", "grid_coord", "index"),
                    feat_keys=("coord", "normal"),
                ),
            ],
            aug_transform=[
                # 1. 基准：无变换
                [
                    dict(type="RandomRotateTargetAngle", angle=[0], axis="z", p=1)
                ],
                
                # 2. 90度旋转（室内场景标准）
                [
                    dict(type="RandomRotateTargetAngle", angle=[1.57], axis="z", p=1)
                ],
                
                # 3. X轴翻转（左右镜像）
                [
                    dict(type="RandomFlip", p=1)
                ],
                
                # 5. 微小缩放（模拟尺度模糊）
                [
                    dict(type="RandomScale", scale=[0.98, 1.02])
                ],
                
                # 6. 微小旋转 + 缩放（模拟边界抖动）
                [
                    dict(type="RandomRotate", angle=[-0.1, 0.1], axis="z", center=[0, 0, 0], p=1),
                    dict(type="RandomScale", scale=[0.99, 1.01])
                ],
                
                # 7. 组合：90度 + 微缩放
                [
                    dict(type="RandomRotateTargetAngle", angle=[1.57], axis="z", p=1),
                    dict(type="RandomScale", scale=[0.98, 1.02])
                ],
                
                # 8. 组合：X翻转 + 微旋转
                [
                    dict(type="RandomFlip", p=1),
                    dict(type="RandomRotate", angle=[-0.05, 0.05], axis="z", center=[0, 0, 0], p=1)
                ]
            ],
        ),
    ),
)


# hook
hooks = [
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="SemSegEvaluator"),
    dict(type="CheckpointSaver", save_freq=None),
    dict(type="PreciseEvaluator", test_last=True),
]
train = dict(type='DefaultTrainer')
test = dict(type='SemSegTester', verbose=True)