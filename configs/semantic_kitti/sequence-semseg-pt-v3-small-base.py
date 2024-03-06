_base_ = ["../_base_/default_runtime.py"]

# misc custom setting
max_training_input_points = 200000
training_grid_size = 0.2
validation_grid_size = 0.2
batch_size = 3  # bs: total bs in all gpus
mix_prob = 0.8
empty_cache = True
enable_amp = False

# model settings
model = dict(
    type="DefaultSegmentorV2",
    num_classes=19,
    backbone_out_channels=64,
    backbone=dict(
        type="PT-v3m1",
        in_channels=5,
        order=["z", "z-trans", "hilbert", "hilbert-trans"],
        stride=(2, 2),
        enc_depths=(2, 2, 4),
        enc_channels=(32, 64, 128),
        enc_num_head=(2, 4, 8),
        enc_patch_size=(1024, 1024, 1024),
        dec_depths=(2, 2),
        dec_channels=(64, 128),
        dec_num_head=(4, 8),
        dec_patch_size=(1024, 1024),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=True,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        shuffle_orders=True,
        pre_norm=True,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=False,
        upcast_softmax=False,
        cls_mode=False,
        pdnorm_bn=False,
        pdnorm_ln=False,
        pdnorm_decouple=False,
        pdnorm_adaptive=False,
        pdnorm_affine=False,
        pdnorm_conditions=("ScanNet", "S3DIS", "Structured3D"),
    ),
    criteria=[
        dict(
            type='CrossEntropyLoss',
            weight=[
                3.1557, 8.7029, 7.8281, 6.1354, 6.3161, 7.9937, 8.9704,
                10.1922, 1.6155, 4.2187, 1.9385, 5.5455, 2.0198, 2.6261,
                1.3212, 5.1102, 2.5492, 5.8585, 7.3929
            ],
            loss_weight=1.0,
            ignore_index=-1),
        # dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0, ignore_index=-1),
    ],
)

# scheduler settings
epoch = 25
eval_epoch = 25
optimizer = dict(type="AdamW", lr=0.0004, weight_decay=0.005)
scheduler = dict(
    type="OneCycleLR",
    max_lr=optimizer["lr"],
    pct_start=0.02,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=100.0,
)

# dataset settings
dataset_type = "SemanticKITTIDataset"
data_root = "data/semantic_kitti"
ignore_index = -1
concatenate_scans = True
stack_scans = False
sequence_length = 5
names = [
    "car",
    "bicycle",
    "motorcycle",
    "truck",
    "other-vehicle",
    "person",
    "bicyclist",
    "motorcyclist",
    "road",
    "parking",
    "sidewalk",
    "other-ground",
    "building",
    "fence",
    "vegetation",
    "trunk",
    "terrain",
    "pole",
    "traffic-sign",
]

data = dict(
    num_classes=19,
    ignore_index=ignore_index,
    names=names,
    train=dict(
        type=dataset_type,
        split="train",
        data_root=data_root,
        transform=[
            # dict(type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=0.2),
            # dict(type="RandomRotateTargetAngle", angle=(1/2, 1, 3/2), center=[0, 0, 0], axis="z", p=0.75),
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
            # dict(type="RandomRotate", angle=[-1/6, 1/6], axis="x", p=0.5),
            # dict(type="RandomRotate", angle=[-1/6, 1/6], axis="y", p=0.5),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            # dict(type="RandomShift", shift=[0.2, 0.2, 0.2]),
            dict(type="RandomFlip", p=0.5),
            dict(type="RandomJitter", sigma=0.005, clip=0.02),
            # dict(type="ElasticDistortion", distortion_params=[[0.2, 0.4], [0.8, 1.6]]),
            dict(
                type="GridSample",
                grid_size=0.2,
                hash_type="fnv",
                mode="train",
                keys=("coord", "strength", "segment", "time"),
                return_grid_coord=True,
            ),
            dict(type="PointClip", point_cloud_range=(-35.2, -35.2, -4, 35.2, 35.2, 2)),
            dict(type="SphereCrop", sample_rate=0.8, mode="random"),
            dict(type="SphereCrop", point_max=max_training_input_points, mode="random"),
            # dict(type="CenterShift", apply_z=False),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment"),
                feat_keys=("coord", "strength", "time"),
            ),
        ],
        test_mode=False,
        ignore_index=ignore_index,
    ),
    val=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        transform=[
            dict(
                type="GridSample",
                grid_size=training_grid_size,
                hash_type="fnv",
                mode="train",
                keys=("coord", "strength", "segment", "time"),
                return_grid_coord=True,
            ),
            dict(type="PointClip", point_cloud_range=(-35.2, -35.2, -4, 35.2, 35.2, 2)),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment"),
                feat_keys=("coord", "strength", "time"),
            ),
        ],
        test_mode=False,
        ignore_index=ignore_index,
    ),
    test=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        transform=[],
        test_mode=True,
        test_cfg=dict(
            voxelize=dict(
                type="GridSample",
                grid_size=0.05,
                hash_type="fnv",
                mode="test",
                return_grid_coord=True,
                keys=("coord", "strength", "time"),
            ),
            crop=None,
            post_transform=[
                dict(
                    type="PointClip",
                    point_cloud_range=(-35.2, -35.2, -4, 35.2, 35.2, 2),
                ),
                dict(type="ToTensor"),
                dict(
                    type="Collect",
                    keys=("coord", "grid_coord", "index"),
                    feat_keys=("coord", "strength", "time"),
                ),
            ],
            aug_transform=[
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[0],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    )
                ],
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[1 / 2],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    )
                ],
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[1],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    )
                ],
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[3 / 2],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    )
                ],
            ],
        ),
        ignore_index=ignore_index,
    ),
)
