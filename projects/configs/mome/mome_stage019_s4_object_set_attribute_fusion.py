"""Stage019-S4 frozen-MoME object-set attribute-fusion declaration.

This config inherits the Stage019 AQR probe identity but deliberately does not
insert the S4 module into ``model`` or alter ``simple_test``.  The S4 worker
must instantiate and call the object-set fusion module explicitly after the
no-GT full-context bundle has been decoded.
"""

import math


_base_ = ["./mome_qta_aqr_camera_lidar.py"]

stage019_s4_protocol_profile = "stage019_s4_object_set_attribute_fusion_v1"

object_set_attribute_fusion = dict(
    type="ObjectSetAttributeFusion",
    enabled=False,
    default_inference_mode=None,
    explicit_modes=("score_only", "attribute_only", "stacked"),
    route_order=("fused", "lidar", "camera"),
    feature_version="stage019_s4_deployment_visible_object_features_v1",
    feature_dim=41,
    checkpoint_state_dict_key="object_set_attribute_fusion_state_dict",
    constructor=dict(
        hidden_dim=64,
        max_match_distance=2.0,
        max_detections=300,
        max_rescue_boxes=20,
        max_center_delta=1.0,
        max_dim_log_delta=0.20,
        max_yaw_delta=math.pi / 9.0,
        max_velocity_delta=2.0,
        max_score_logit_residual=1.0,
        max_abs_log_temperature=math.log(2.0),
        rescue_enabled=False,
    ),
    inference_contract=dict(
        gt_forbidden=True,
        fault_name_forbidden=True,
        affected_object_mask_forbidden=True,
        corruption_identity_forbidden=True,
        complete_zero_original_only=True,
        no_match_identity=True,
        nonfinite_fail_closed=True,
        anchor_label_preserved=True,
        output_box_cap=300,
        rescue_requires_explicit_enable=True,
        rescue_box_cap=20,
        rescue_implementation_status=(
            "scientifically_disabled_until_union_gate_and_trained_rescue_checkpoint"
        ),
    ),
)
