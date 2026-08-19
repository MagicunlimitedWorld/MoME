"""Fail-closed contracts for Stage019-S4 object-set attribute fusion.

This module deliberately has no MMDetection dependency.  Workers, trainers,
mergers and the controller import the same constants and gate functions so a
scientific threshold cannot drift between execution phases.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


STAGE_ID = "Stage019-S4"
PROTOCOL_PROFILE = "stage019_s4_object_set_attribute_fusion_v1"
RUN_ID = "2026-08-20-mome-stage019-s4-object-set-attribute-fusion-v1"

ACTIONABLE_CONDITIONS = (
    "clean",
    "beam_reduction_4",
    "limited_fov_original_code_60",
    "lidar_object_failure",
    "camera_mud_mask",
)
FAULT_CONDITIONS = ACTIONABLE_CONDITIONS[1:]
HARD_BYPASS_CONDITIONS = ("lidar_zero", "camera_zero")
ALL_CONDITIONS = (
    "clean",
    "beam_reduction_4",
    "lidar_zero",
    "limited_fov_original_code_60",
    "lidar_object_failure",
    "camera_zero",
    "camera_mud_mask",
)
EXPERT_ROLES = ("fused", "lidar", "camera")

PRIMARY_SEED = 20260710
CONFIRMATION_SEED = 20260711
TRAINING20_INPUTS_MANIFEST_SHA256 = (
    "0787A4DB8568DAAD35B8B88DB34E126B0443F000CE93F94FC776A2914FC49910"
)
SCENE_SPLIT_SHA256 = (
    "0C1C9581B128FFFEB03E456846FA12548498F5A1494EFC03FCFE46C85567DF97"
)
TRAIN_ANNOTATION_SHA256 = (
    "9214A60ADF903647DF027FE2F49CFBBB3CDF14DC4CE2C0B099337F7E92AB34A6"
)
SAMPLE_SCENE_MAP_SHA256 = (
    "D9D129B3600AC57F30A7B978D6E251AFA84C01A68646EC8BF1B9BE011680EB24"
)
VALIDATION_BEAM4_ANNOTATION_SHA256 = (
    "3BEFE714F92E7E864C308A0A4B9A2BC4008F82FEBF1888AA680DE7BB2F90F0A5"
)

PILOT_SCENES = (
    "fcc020250f884397965ba00c1d9ad9e6",
    "3ada261efee347cba2e7557794f1aec8",
    "efa5c96f05594f41a2498eb9f2e7ad99",
    "fcbccedd61424f1b85dcbf8f897f9754",
    "16e50a63b809463099cb4c378fe0641e",
    "64bfc5edd71147858ce7446892d7f864",
    "380ff00ec86447e3b986edc8e82ffba7",
    "a04daf2d0f194b2ab2ff2a47dfebc1d7",
    "96e5f1f0944946f391b4ef33ad623008",
    "e036014a715945aa965f4ec24e8639c9",
)
PILOT_SCENE_SERIALIZATION_SHA256 = (
    "375A20E4264E55BAEDE411A069979F6E57306EF7C0B7B1B60843C28A1349679D"
)

G0_SCENE_COUNT = 20
G0_FRAME_COUNT = 803
G0_FRAME_CONDITION_COUNT = 4015
G0_FOLD_SIZES = (7, 7, 6)
FIT_SCENE_COUNT = 80
CALIBRATION_SCENE_COUNT = 20
PILOT_SCENE_COUNT = 10
PILOT_FRAME_COUNT = 401
FULLVAL_SCENE_COUNT = 150
FULLVAL_FRAME_COUNT = 6019
MAX_OUTPUT_BOXES = 300
MAX_RESCUE_BOXES = 20
MAX_TRAINABLE_PARAMETERS = 50_000

MATCH_RADIUS_METERS = 2.0
SOFT_LABEL_THRESHOLDS_METERS = (0.5, 1.0, 2.0, 4.0)
FUSION_STRENGTH_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
SINGLE_FAULT_HARM_FLOOR = -0.005


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def canonical_scene_serialization(scene_tokens: Sequence[str]) -> bytes:
    """UTF-8, LF, no trailing LF; order and duplicates are significant."""

    values = [str(value) for value in scene_tokens]
    if any(not value for value in values):
        raise ValueError("scene token serialization contains an empty token")
    if len(values) != len(set(values)):
        raise ValueError("scene token serialization contains duplicates")
    return "\n".join(values).encode("utf-8")


def scene_serialization_sha256(scene_tokens: Sequence[str]) -> str:
    return hashlib.sha256(canonical_scene_serialization(scene_tokens)).hexdigest().upper()


def validate_pilot_identity(scene_tokens: Sequence[str], frame_count: int) -> None:
    if tuple(str(value) for value in scene_tokens) != PILOT_SCENES:
        raise ValueError("pilot scene order differs from the preregistered 10 scenes")
    observed = scene_serialization_sha256(scene_tokens)
    if observed != PILOT_SCENE_SERIALIZATION_SHA256:
        raise ValueError("pilot scene serialization SHA256 mismatch")
    if int(frame_count) != PILOT_FRAME_COUNT:
        raise ValueError("pilot must contain exactly 401 frames")


def split_g0_folds(scene_tokens: Sequence[str]) -> tuple[tuple[str, ...], ...]:
    values = tuple(str(value) for value in scene_tokens)
    if len(values) != G0_SCENE_COUNT or len(set(values)) != G0_SCENE_COUNT:
        raise ValueError("G0 requires exactly 20 unique scenes")
    folds = []
    offset = 0
    for size in G0_FOLD_SIZES:
        folds.append(values[offset : offset + size])
        offset += size
    if tuple(len(fold) for fold in folds) != G0_FOLD_SIZES or offset != len(values):
        raise AssertionError("internal G0 fold construction drifted")
    return tuple(folds)


def validate_train_scene_split(
    fit_scenes: Sequence[str],
    calibration_scenes: Sequence[str],
    training20_scenes: Sequence[str],
) -> dict[str, Any]:
    fit = tuple(str(value) for value in fit_scenes)
    calibration = tuple(str(value) for value in calibration_scenes)
    training20 = tuple(str(value) for value in training20_scenes)
    if len(fit) != FIT_SCENE_COUNT or len(set(fit)) != FIT_SCENE_COUNT:
        raise ValueError("fit split must contain exactly 80 unique scenes")
    if (
        len(calibration) != CALIBRATION_SCENE_COUNT
        or len(set(calibration)) != CALIBRATION_SCENE_COUNT
    ):
        raise ValueError("calibration split must contain exactly 20 unique scenes")
    if len(training20) != G0_SCENE_COUNT or len(set(training20)) != G0_SCENE_COUNT:
        raise ValueError("training20 must contain exactly 20 unique scenes")
    overlap = set(fit) & set(calibration)
    if overlap:
        raise ValueError("fit80 and calibration20 scene sets overlap")
    missing = set(training20) - set(fit)
    if missing:
        raise ValueError("training20 is not a subset of fit80")
    if set(training20) & set(calibration):
        raise ValueError("training20 overlaps calibration20")
    return {
        "fit_scene_count": len(fit),
        "calibration_scene_count": len(calibration),
        "training20_scene_count": len(training20),
        "fit_calibration_overlap_count": 0,
        "training20_is_fit_subset": True,
        "training20_calibration_overlap_count": 0,
    }


@dataclass(frozen=True)
class MetricDelta:
    mAP: float
    NDS: float

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "MetricDelta":
        output = cls(mAP=float(value["mAP"]), NDS=float(value["NDS"]))
        if not math.isfinite(output.mAP) or not math.isfinite(output.NDS):
            raise ValueError("metric delta must be finite")
        return output

    def as_dict(self) -> dict[str, float]:
        return {"mAP": self.mAP, "NDS": self.NDS}


def _exact_condition_metric_map(
    values: Mapping[str, Mapping[str, Any]],
    conditions: Sequence[str],
) -> dict[str, MetricDelta]:
    if set(values) != set(conditions):
        raise ValueError(
            "metric condition set mismatch: expected={}, observed={}".format(
                sorted(conditions), sorted(values)
            )
        )
    return {condition: MetricDelta.from_mapping(values[condition]) for condition in conditions}


def metric_macro(
    values: Mapping[str, MetricDelta], conditions: Sequence[str]
) -> MetricDelta:
    if not conditions:
        raise ValueError("macro metric requires at least one condition")
    return MetricDelta(
        mAP=sum(values[condition].mAP for condition in conditions) / len(conditions),
        NDS=sum(values[condition].NDS for condition in conditions) / len(conditions),
    )


def evaluate_g0_gate(
    score_deltas: Mapping[str, Mapping[str, Any]],
    attribute_minus_score: Mapping[str, Mapping[str, Any]],
    union_minus_attribute: Mapping[str, Mapping[str, Any]],
    score_fold_fault_macro: Sequence[Mapping[str, Any]],
    attribute_fold_fault_macro: Sequence[Mapping[str, Any]],
    union_unique_opportunity_conditions: Iterable[str],
) -> dict[str, Any]:
    """Evaluate all preregistered G0 promotion decisions without fallbacks."""

    score = _exact_condition_metric_map(score_deltas, FAULT_CONDITIONS)
    attribute = _exact_condition_metric_map(attribute_minus_score, FAULT_CONDITIONS)
    union = _exact_condition_metric_map(union_minus_attribute, FAULT_CONDITIONS)
    score_folds = [MetricDelta.from_mapping(value) for value in score_fold_fault_macro]
    attribute_folds = [MetricDelta.from_mapping(value) for value in attribute_fold_fault_macro]
    if len(score_folds) != 3 or len(attribute_folds) != 3:
        raise ValueError("G0 gates require the preregistered 7/7/6 three-fold metrics")
    score_macro = metric_macro(score, FAULT_CONDITIONS)
    attribute_macro = metric_macro(attribute, FAULT_CONDITIONS)
    union_macro = metric_macro(union, FAULT_CONDITIONS)
    score_positive_folds = sum(value.mAP > 0.0 for value in score_folds)
    # The 0.0025 threshold belongs to the aggregate macro gate.  "Same
    # direction" at fold level means strictly positive NDS; applying 0.0025 to
    # every fold would silently add a stricter, non-preregistered gate.
    attribute_same_direction_folds = sum(value.NDS > 0.0 for value in attribute_folds)
    score_pass = bool(
        score_macro.mAP >= 0.005
        and score_macro.NDS >= 0.0
        and score_positive_folds >= 2
    )
    attribute_joint_conditions = [
        condition
        for condition in FAULT_CONDITIONS
        if attribute[condition].mAP >= 0.005 and attribute[condition].NDS >= 0.0025
    ]
    attribute_pass = bool(
        attribute_macro.NDS >= 0.0025
        and attribute_joint_conditions
        and attribute_same_direction_folds >= 2
    )
    opportunity = tuple(sorted(set(str(value) for value in union_unique_opportunity_conditions)))
    unknown = set(opportunity) - set(FAULT_CONDITIONS)
    if unknown:
        raise ValueError("G0-union opportunity condition is not actionable")
    # Opportunity coverage is reported diagnostically.  The preregistered
    # rescue authorization criterion is only the four-fault macro mAP delta.
    rescue_pass = bool(union_macro.mAP >= 0.005)
    base_training_authorized = bool(score_pass or attribute_pass)
    training_authorized = base_training_authorized
    return {
        "schema": "visfuse3d_stage019_s4_g0_gate_decision_v1",
        "protocol_profile": PROTOCOL_PROFILE,
        "score": {
            "passed": score_pass,
            "fault_macro_delta": score_macro.as_dict(),
            "positive_map_fold_count": score_positive_folds,
            "required_positive_map_fold_count": 2,
        },
        "attribute": {
            "passed": attribute_pass,
            "fault_macro_delta_vs_score": attribute_macro.as_dict(),
            "jointly_improved_conditions": attribute_joint_conditions,
            "same_direction_fold_count": attribute_same_direction_folds,
            "required_same_direction_fold_count": 2,
        },
        "union_rescue": {
            "passed": rescue_pass,
            "fault_macro_delta_vs_attribute": union_macro.as_dict(),
            "unique_opportunity_conditions": list(opportunity),
            "max_added_boxes_per_frame": MAX_RESCUE_BOXES,
            "implementation_status": (
                "conditional_implementation_required_before_training"
                if rescue_pass
                else "not_authorized_by_g0_union"
            ),
        },
        "base_small_head_training_gate_passed": base_training_authorized,
        "training_authorized": training_authorized,
        "g1_authorized": score_pass,
        "g2_authorized": attribute_pass,
        "rescue_authorized": rescue_pass,
        "rescue_implementation_required": rescue_pass,
        "status": (
            "passed_base_heads_rescue_implementation_required_before_rescue_evaluation"
            if base_training_authorized and rescue_pass
            else (
                "passed_at_least_one_trainable_branch"
                if base_training_authorized
                else "stopped_g0_scientific_gate_failed"
            )
        ),
    }


def choose_g1_strength(
    deltas_by_strength: Mapping[float, Mapping[str, Mapping[str, Any]]]
) -> dict[str, Any]:
    """Choose the smallest tied strength after applying calibration safety."""

    normalized = {float(key): value for key, value in deltas_by_strength.items()}
    if set(normalized) != set(FUSION_STRENGTH_GRID):
        raise ValueError("G1 calibration must evaluate the exact five fusion strengths")
    candidates = []
    diagnostics = {}
    for strength in FUSION_STRENGTH_GRID:
        metrics = _exact_condition_metric_map(normalized[strength], ACTIONABLE_CONDITIONS)
        fault_macro = metric_macro(metrics, FAULT_CONDITIONS)
        safe = bool(
            metrics["clean"].mAP >= 0.0
            and metrics["clean"].NDS >= 0.0
            and fault_macro.NDS >= 0.0
        )
        diagnostics[str(strength)] = {
            "safe": safe,
            "clean_delta": metrics["clean"].as_dict(),
            "fault_macro_delta": fault_macro.as_dict(),
        }
        if safe:
            candidates.append((fault_macro.mAP, -strength, strength))
    if not candidates:
        return {
            "status": "stopped_no_safe_fusion_strength",
            "passed": False,
            "selected_strength": None,
            "grid": diagnostics,
        }
    _, _, selected = max(candidates)
    selected_metrics = _exact_condition_metric_map(
        normalized[selected], ACTIONABLE_CONDITIONS
    )
    selected_macro = metric_macro(selected_metrics, FAULT_CONDITIONS)
    passed = selected > 0.0 and selected_macro.mAP > 0.0 and selected_macro.NDS >= 0.0
    return {
        "status": "passed" if passed else "stopped_g1_no_positive_selected_strength",
        "passed": passed,
        "selected_strength": selected,
        "selected_fault_macro_delta": selected_macro.as_dict(),
        "grid": diagnostics,
    }


def evaluate_validation_gate(
    deltas: Mapping[str, Mapping[str, Any]],
    changed_object_count: int,
    accepted_object_count: int,
    zero_identity: Mapping[str, Mapping[str, Any]],
    method: str,
    attribute_vs_score_nds: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    values = _exact_condition_metric_map(deltas, ACTIONABLE_CONDITIONS)
    if set(zero_identity) != set(HARD_BYPASS_CONDITIONS):
        raise ValueError("validation gate requires both complete-zero identities")
    zero_checks = {}
    for condition, evidence in zero_identity.items():
        original = evidence.get("original_prediction", {})
        final = evidence.get("final_prediction", {})
        original_path = Path(str(original.get("path", "")))
        final_path = Path(str(final.get("path", "")))
        original_sha = str(original.get("sha256", "")).upper()
        final_sha = str(final.get("sha256", "")).upper()
        zero_checks[condition] = bool(
            evidence.get("same_run_original_reference")
            and original_path.is_file()
            and final_path.is_file()
            and len(original_sha) == 64
            and original_sha == final_sha
            and sha256_file(original_path) == original_sha
            and sha256_file(final_path) == final_sha
        )
    macro = metric_macro(values, FAULT_CONDITIONS)
    jointly_improved = [
        condition
        for condition in FAULT_CONDITIONS
        if values[condition].mAP > 0.0 and values[condition].NDS > 0.0
    ]
    harmed = [
        condition
        for condition in FAULT_CONDITIONS
        if values[condition].mAP < SINGLE_FAULT_HARM_FLOOR
        or values[condition].NDS < SINGLE_FAULT_HARM_FLOOR
    ]
    checks = {
        "clean_map_nds_non_decreasing": (
            values["clean"].mAP >= 0.0 and values["clean"].NDS >= 0.0
        ),
        "four_fault_macro_map_nds_strictly_positive": (
            macro.mAP > 0.0 and macro.NDS > 0.0
        ),
        "at_least_one_fault_jointly_improved": bool(jointly_improved),
        "no_fault_below_minus_0p005": not harmed,
        "actual_changed_object_count_nonzero": int(changed_object_count) > 0,
        "actual_accepted_object_count_nonzero": int(accepted_object_count) > 0,
        "complete_zero_prediction_sha256_identity": all(zero_checks.values()),
    }
    attribute_gain_conditions = []
    if method == "cp_afr":
        if attribute_vs_score_nds is None or set(attribute_vs_score_nds) != set(FAULT_CONDITIONS):
            raise ValueError("CP-AFR gate requires per-fault NDS delta versus score baseline")
        attribute_gain_conditions = [
            condition
            for condition in FAULT_CONDITIONS
            if math.isfinite(float(attribute_vs_score_nds[condition]))
            and float(attribute_vs_score_nds[condition]) >= 0.0025
        ]
        checks["cp_afr_one_fault_nds_plus_0p0025_vs_score"] = bool(
            attribute_gain_conditions
        )
    elif method != "gace_lite":
        raise ValueError("method must be gace_lite or cp_afr")
    passed = all(checks.values())
    return {
        "schema": "visfuse3d_stage019_s4_validation_gate_decision_v1",
        "protocol_profile": PROTOCOL_PROFILE,
        "method": method,
        "status": "passed" if passed else "stopped_scientific_gate_failed",
        "passed": passed,
        "checks": checks,
        "fault_macro_delta": macro.as_dict(),
        "jointly_improved_fault_conditions": jointly_improved,
        "harmed_fault_conditions": harmed,
        "cp_afr_gain_conditions": attribute_gain_conditions,
        "changed_object_count": int(changed_object_count),
        "accepted_object_count": int(accepted_object_count),
        "zero_identity_checks": zero_checks,
    }


def validate_merge_coverage(
    manifests: Sequence[Mapping[str, Any]],
    expected_scene_count: int,
    expected_frame_count: int,
) -> dict[str, int]:
    if len(manifests) != 2:
        raise ValueError("S4 merge requires exactly two scene-shard manifests")
    scenes: set[str] = set()
    tokens: set[str] = set()
    for manifest in manifests:
        if manifest.get("status") != "complete" or int(
            manifest.get("engineering_failure_count", -1)
        ) != 0:
            raise ValueError("S4 shard is not complete and engineering-clean")
        shard_scenes = [str(value) for value in manifest.get("scene_tokens", [])]
        shard_tokens = [str(value) for value in manifest.get("frame_tokens", [])]
        if len(shard_scenes) != len(set(shard_scenes)):
            raise ValueError("S4 shard contains duplicate scenes")
        if len(shard_tokens) != len(set(shard_tokens)):
            raise ValueError("S4 shard contains duplicate frames")
        if scenes & set(shard_scenes) or tokens & set(shard_tokens):
            raise ValueError("S4 shards overlap")
        scenes.update(shard_scenes)
        tokens.update(shard_tokens)
    if len(scenes) != int(expected_scene_count) or len(tokens) != int(expected_frame_count):
        raise ValueError("S4 merged coverage does not match preregistered counts")
    return {
        "scene_count": len(scenes),
        "frame_count": len(tokens),
        "unique_token_count": len(tokens),
    }


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest().upper()
