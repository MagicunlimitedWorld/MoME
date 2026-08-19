"""MoME-QTA-AQR probe/inference config.

The detector remains frozen.  Stage019-B trains the 256->3 head from cached
train-only router features and route-loss advantages with
``tools/qta/train_qta_head.py``.  Supply the frozen canonical MoME checkpoint,
the trained QTA-head checkpoint and calibrated thresholds through the control
runner; no private workstation path is committed here.
"""

_base_ = ['./mome.py']

qta_seed = 20260710
point_cloud_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone',
]
img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675],
    std=[57.375, 57.120, 58.395],
    to_rgb=False,
)
ida_aug_conf = {
    'resize_lim': (0.94, 1.25),
    'final_dim': (640, 1600),
    'bot_pct_lim': (0.0, 0.0),
    'rot_lim': (0.0, 0.0),
    'H': 900,
    'W': 1600,
    'rand_flip': True,
}

qta_train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=[0, 1, 2, 3, 4],
    ),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=10,
        use_dim=[0, 1, 2, 3, 4],
    ),
    dict(type='LoadMultiViewImageFromFiles'),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(
        type='GlobalRotScaleTransAll',
        rot_range=[-0.3925 * 2, 0.3925 * 2],
        scale_ratio_range=[0.9, 1.1],
        translation_std=[0.5, 0.5, 0.5],
    ),
    dict(
        type='CustomRandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5,
    ),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='PointShuffle'),
    dict(
        type='ResizeCropFlipImage',
        data_aug_conf=ida_aug_conf,
        training=True,
    ),
    dict(type='QtaLocalCorruption3D', mode='train', seed=qta_seed),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(
        type='Collect3D',
        keys=['points', 'img', 'gt_bboxes_3d', 'gt_labels_3d'],
        meta_keys=(
            'filename', 'ori_shape', 'img_shape', 'lidar2img', 'depth2img',
            'cam2img', 'pad_shape', 'scale_factor', 'flip',
            'pcd_horizontal_flip', 'pcd_vertical_flip', 'box_mode_3d',
            'box_type_3d', 'img_norm_cfg', 'pcd_trans', 'sample_idx',
            'pcd_scale_factor', 'pcd_rotation', 'pts_filename',
            'transformation_3d_flow', 'rot_degree', 'gt_bboxes_3d',
            'gt_labels_3d', 'modalmask', 'qta_corruption',
            'qta_route_available', 'qta_hard_bypass',
        ),
    ),
]

data = dict(
    train=dict(
        dataset=dict(
            pipeline=qta_train_pipeline,
        ),
    ),
)

model = dict(
    pts_bbox_head=dict(
        transformer=dict(
            qta_router=dict(
                type='QueryTaskAdvantageRouter',
                input_dim=256,
                thresholds=[float('inf'), float('inf'), float('inf')],
                max_overrides_per_frame=0,
                enabled=False,
                checkpoint=None,
            ),
        ),
    ),
)

# The control runner must set this to the frozen canonical ``mome`` checkpoint.
load_from = None
total_epochs = 3

custom_hooks = [
    dict(
        type='FreezeWeight',
        finetune_weight=[
            'pts_bbox_head.transformer.qta_router.advantage_head',
        ],
    ),
]

qta_identity = dict(
    canonical_model_id='mome_qta_aqr_camera_lidar',
    display_name='MoME-QTA-AQR (Camera+LiDAR)',
    frozen_baseline_canonical_model_id='mome',
    frozen_baseline_display_name='MoME',
)

qta_data_contract = dict(
    split_seed=qta_seed,
    proper_training_scenes=560,
    calibration_scenes=140,
    formal_nuscenes_r_inputs_forbidden=True,
    object_paste=False,
)
