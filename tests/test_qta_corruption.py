from __future__ import annotations

import numpy as np
import torch
from mmdet3d.core.bbox import LiDARInstance3DBoxes
from mmdet3d.core.points import LiDARPoints

from projects.mmdet3d_plugin.datasets.pipelines.transform_3d import (
    QtaLocalCorruption3D,
)


def synthetic_sample() -> dict[str, object]:
    return {
        "sample_idx": "synthetic-token",
        "pts_filename": "D:/project/nuscenes/samples/LIDAR_TOP/a.bin",
        "filename": [
            f"D:/project/nuscenes/samples/CAM_FRONT/{index}.jpg"
            for index in range(6)
        ],
        "points": LiDARPoints(
            torch.tensor(
                [
                    [1.0, 0.0, 0.0, 0.5, 0.0],
                    [2.0, 1.0, 0.0, 0.5, 0.0],
                    [-1.0, -1.0, 0.0, 0.5, 0.0],
                ]
            ),
            points_dim=5,
        ),
        "gt_bboxes_3d": LiDARInstance3DBoxes(
            torch.tensor(
                [[1.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0]]
            ),
            box_dim=9,
        ),
        "img": [np.full((64, 96, 3), 127, dtype=np.uint8) for _ in range(6)],
    }


def test_object_point_corruption_supports_cpu_nuscenes_9d_boxes() -> None:
    result = QtaLocalCorruption3D(
        forced_condition="lidar_object_points_missing"
    )(synthetic_sample())
    corruption = result["qta_corruption"]
    assert corruption["condition"] == "lidar_object_points_missing"
    assert corruption["details"]["selected_box_indices"] == [0]


def test_complete_zero_conditions_set_hard_bypass() -> None:
    lidar = QtaLocalCorruption3D(forced_condition="lidar_zero")(
        synthetic_sample()
    )
    assert lidar["qta_hard_bypass"] is True
    assert lidar["qta_route_available"] == [False, False, True]
    assert torch.count_nonzero(lidar["points"].tensor).item() == 0

    camera = QtaLocalCorruption3D(forced_condition="camera_zero")(
        synthetic_sample()
    )
    assert camera["qta_hard_bypass"] is True
    assert camera["qta_route_available"] == [False, True, False]
    assert sum(np.count_nonzero(image) for image in camera["img"]) == 0
