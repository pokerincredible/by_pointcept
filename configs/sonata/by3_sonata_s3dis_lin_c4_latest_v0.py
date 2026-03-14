_base_ = ["../_base_/default_runtime.py"]

# ==============================
# runtime
# ==============================

batch_size = 8
num_worker = 12
mix_prob = 0.8

epoch = 3000
eval_epoch = 3000

enable_amp = True
clip_grad = 3.0


# ==============================
# optimizer
# ==============================

optimizer = dict(
    type="AdamW",
    lr=0.001,
    weight_decay=0.02
)

scheduler = dict(
    type="OneCycleLR",
    max_lr=0.0015,
    pct_start=0.1,
    anneal_strategy="cos",
    div_factor=10,
    final_div_factor=1000
)


# ==============================
# model
# ==============================

model = dict(
    type="DefaultSegmentorV2",
    num_classes=4,
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
        drop_path=0.15,
        enable_flash=True,
        shuffle_orders=True,
        enc_mode=True,
        mask_token=False,
        traceable=False,
    ),

    criteria=[

        dict(
            type="CrossEntropyLoss",
            loss_weight=1.0,
            ignore_index=-1,
            label_smoothing=0.05,
            weight=[1.2,1.0,3.0,2.5]
        ),

        dict(
            type="LovaszLoss",
            mode="multiclass",
            loss_weight=1.0,
            ignore_index=-1
        )
    ]
)


# ==============================
# dataset
# ==============================

dataset_type = "S3DISDataset"

data_root = "/home/poker/workspace/11_third_projs/data/train_data/01_version/c4/c4_dataset"

data = dict(

    num_classes=4,
    ignore_index=-1,

    names=["class0","class1","class2","class3"],

    train=dict(
        type=dataset_type,

        split=("Area_1","Area_patch_rare","Area_patch_boundary","Area_patch_hard"),

        data_root=data_root,

        transform=[

            dict(type="RandomRotate",angle=[-1,1],axis="z",p=0.5),

            dict(type="RandomRotate",angle=[-0.05,0.05],axis="x",p=0.5),

            dict(type="RandomRotate",angle=[-0.05,0.05],axis="y",p=0.5),

            dict(type="RandomScale",scale=[0.9,1.1]),

            dict(type="RandomFlip",p=0.5),

            dict(type="RandomJitter",sigma=0.6,clip=1.2),

            dict(
                type="GridSample",
                grid_size=0.48,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True
            ),

            dict(type="ToTensor"),

            dict(
                type="Collect",
                keys=("coord","grid_coord","segment"),
                feat_keys=("coord","normal")
            )
        ],

        test_mode=False,
        loop=1
    ),

    val=dict(
        type=dataset_type,
        split="Area_5",
        data_root=data_root,

        transform=[

            dict(
                type="GridSample",
                grid_size=0.48,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True
            ),

            dict(type="ToTensor"),

            dict(
                type="Collect",
                keys=("coord","grid_coord","segment"),
                feat_keys=("coord","normal")
            )
        ],

        test_mode=False
    )
)


# ==============================
# hooks
# ==============================

hooks = [

    dict(type="IterationTimer"),

    dict(type="InformationWriter"),

    dict(type="SemSegEvaluator"),

    dict(type="CheckpointSaver"),

    dict(type="PreciseEvaluator",test_last=True)
]


train = dict(type="DefaultTrainer")

test = dict(type="SemSegTester",verbose=True)