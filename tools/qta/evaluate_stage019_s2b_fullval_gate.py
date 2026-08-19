"""Evaluate the preregistered gate for Stage019-S2-B full validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from .common import atomic_write_json
    from .extract_route_loss_cache import is_relative_to, sha256_file
    from .full_query_oracle_worker import CONDITIONS
    from .stage019_s2_training_conditions import (
        S3_ACTIONABLE_CONDITIONS,
        S3_HARD_BYPASS_CONDITIONS,
    )
except ImportError:
    from common import atomic_write_json
    from extract_route_loss_cache import is_relative_to, sha256_file
    from full_query_oracle_worker import CONDITIONS
    from stage019_s2_training_conditions import (
        S3_ACTIONABLE_CONDITIONS,
        S3_HARD_BYPASS_CONDITIONS,
    )


FAULT_CONDITIONS = tuple(value for value in CONDITIONS if value != "clean")
LEGACY_PROTOCOL_PROFILE = "stage019_s2_legacy_v1"
S3_PROTOCOL_PROFILE = "stage019_s3_actionable_hard_bypass_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--margins", type=Path, required=True)
    parser.add_argument("--training-audit-manifest", type=Path, required=True)
    parser.add_argument("--s2a-manifest", type=Path, action="append", required=True)
    parser.add_argument("--pilot-manifest", type=Path, action="append", required=True)
    parser.add_argument("--hard-bypass-manifest", type=Path, action="append", default=[])
    parser.add_argument(
        "--protocol-profile",
        choices=(LEGACY_PROTOCOL_PROFILE, S3_PROTOCOL_PROFILE),
        default=LEGACY_PROTOCOL_PROFILE,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    return parser.parse_args()


def _load_by_condition(
    paths: list[Path],
    mode: str,
    official: bool,
    expected_conditions: tuple[str, ...],
) -> dict[str, dict]:
    if len(paths) != len(expected_conditions):
        raise ValueError("gate requires exactly one manifest for every expected condition")
    output = {}
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        condition = str(payload.get("condition"))
        required_status = "complete_official_evaluation" if official else "complete_diagnostic_merge"
        if (
            payload.get("status") != required_status
            or payload.get("mode") != mode
            or condition not in expected_conditions
            or int(payload.get("engineering_failure_count", -1)) != 0
            or condition in output
        ):
            raise ValueError(f"invalid/duplicate gate manifest: {path}")
        output[condition] = payload
    if set(output) != set(expected_conditions):
        raise ValueError("gate condition set is incomplete")
    return output


def _load_hard_bypass(paths: list[Path]) -> dict[str, dict]:
    if len(paths) != len(S3_HARD_BYPASS_CONDITIONS):
        raise ValueError("S3 gate requires two complete-zero hard-bypass manifests")
    output = {}
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        condition = str(payload.get("condition"))
        base = payload.get("base_prediction", {})
        final = payload.get("final_prediction", {})
        base_path = Path(str(base.get("path", "")))
        final_path = Path(str(final.get("path", "")))
        expected_sha = str(base.get("sha256", "")).upper()
        if (
            payload.get("status") != "complete_hard_bypass_identity"
            or payload.get("protocol_profile") != S3_PROTOCOL_PROFILE
            or payload.get("mode") != "s3b_hard_bypass"
            or payload.get("applicability") != "hard_bypass_not_applicable"
            or condition not in S3_HARD_BYPASS_CONDITIONS
            or condition in output
            or int(payload.get("scene_count", -1)) != 150
            or int(payload.get("frame_count", -1)) != 6019
            or int(payload.get("unique_token_count", -1)) != 6019
            or int(payload.get("engineering_failure_count", -1)) != 0
            or not bool(payload.get("prediction_sha256_identical"))
            or base != final
            or not base_path.is_file()
            or base_path.resolve() != final_path.resolve()
            or sha256_file(base_path).upper() != expected_sha
        ):
            raise ValueError(f"invalid hard-bypass identity manifest: {path}")
        output[condition] = payload
    if set(output) != set(S3_HARD_BYPASS_CONDITIONS):
        raise ValueError("S3 hard-bypass condition set is incomplete")
    return output


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    artifact_root = args.artifact_root.resolve()
    source_root = Path(__file__).resolve().parents[2]
    if not is_relative_to(output_dir, artifact_root):
        raise ValueError("gate output must stay inside the S2 artifact root")
    if is_relative_to(output_dir, source_root):
        raise ValueError("gate output cannot enter the source checkout")
    if output_dir.exists():
        raise FileExistsError(f"immutable S2-B gate exists: {output_dir}")
    output_dir.mkdir(parents=True)
    margins = json.loads(args.margins.read_text(encoding="utf-8"))
    is_s3 = args.protocol_profile == S3_PROTOCOL_PROFILE
    actionable_conditions = S3_ACTIONABLE_CONDITIONS if is_s3 else CONDITIONS
    fault_conditions = (
        tuple(value for value in S3_ACTIONABLE_CONDITIONS if value != "clean")
        if is_s3
        else FAULT_CONDITIONS
    )
    budget = margins.get("candidate_budget", {})
    calibration_gate = bool(
        margins.get("status") == "passed"
        and (
            not is_s3
            or margins.get("protocol_profile") == S3_PROTOCOL_PROFILE
        )
        and (
            not is_s3
            or tuple(margins.get("actionable_conditions", ()))
            == S3_ACTIONABLE_CONDITIONS
        )
        and (
            not is_s3
            or tuple(margins.get("hard_bypass_conditions", ()))
            == S3_HARD_BYPASS_CONDITIONS
        )
        and budget.get("status") == "coverage_reached"
        and int(budget.get("K", -1)) in (4, 8, 16)
        and float(budget.get("coverage", {}).get(str(budget.get("K")), 0.0)) >= 0.95
    )
    training = json.loads(args.training_audit_manifest.read_text(encoding="utf-8"))
    training_gate = bool(
        training.get("status") == "passed_locked_greedy_audit"
        and bool(training.get("locked_greedy_audit"))
        and int(training.get("frame_condition_count", -1))
        == len(actionable_conditions) * 803
        and tuple(training.get("conditions", ())) == tuple(actionable_conditions)
        and int(training.get("accepted_count", 0)) > 0
        and training.get("locked_margins_sha256") == sha256_file(args.margins)
    )
    s2a = _load_by_condition(
        args.s2a_manifest, mode="s2a", official=True, expected_conditions=CONDITIONS
    )
    pilot = _load_by_condition(
        args.pilot_manifest,
        mode="s2b",
        official=False,
        expected_conditions=actionable_conditions,
    )
    hard_bypass = _load_hard_bypass(args.hard_bypass_manifest) if is_s3 else {}
    if not is_s3 and args.hard_bypass_manifest:
        raise ValueError("legacy S2 gate does not accept hard-bypass manifests")
    if is_s3 and any(
        payload.get("protocol_profile") != S3_PROTOCOL_PROFILE
        or int(payload.get("scene_count", -1)) != 150
        or int(payload.get("frame_count", -1)) != 6019
        or int(payload.get("unique_token_count", -1)) != 6019
        for payload in s2a.values()
    ):
        raise ValueError("every S3-A condition must be a complete 150-scene/6,019-token merge")
    if is_s3 and any(
        payload.get("protocol_profile") != S3_PROTOCOL_PROFILE
        for payload in pilot.values()
    ):
        raise ValueError("every S3-B pilot manifest must use the S3 protocol profile")
    if any(
        int(payload.get("scene_count", -1)) != 10
        or int(payload.get("frame_count", -1)) != 401
        for payload in pilot.values()
    ):
        raise ValueError("every S2-B pilot condition must cover locked 10 scenes/401 frames")
    pilot_gate = sum(int(payload.get("accepted_count", 0)) for payload in pilot.values()) > 0

    main_delta = {
        condition: s2a[condition]["delta"]["keep_anchored_minus_original_mome"]
        for condition in CONDITIONS
    }
    clean_gate = bool(
        float(main_delta["clean"]["mAP"]) >= 0.0
        and float(main_delta["clean"]["NDS"]) >= 0.0
    )
    fault_macro = {
        metric: float(
            np.mean([float(main_delta[condition][metric]) for condition in fault_conditions])
        )
        for metric in ("mAP", "NDS")
    }
    macro_gate = fault_macro["mAP"] > 0.0 and fault_macro["NDS"] > 0.0
    jointly_improved = [
        condition
        for condition in fault_conditions
        if float(main_delta[condition]["mAP"]) > 0.0
        and float(main_delta[condition]["NDS"]) > 0.0
    ]
    same_condition_gate = bool(jointly_improved)
    checks = {
        "s2a_clean_map_nds_non_decreasing": clean_gate,
        (
            "four_actionable_fault_macro_map_nds_strictly_positive"
            if is_s3
            else "six_fault_macro_map_nds_strictly_positive"
        ): macro_gate,
        "same_fault_condition_map_nds_strictly_positive": same_condition_gate,
        "calibration_K_nonzero_and_95pct_coverage": calibration_gate,
        "training20_has_locked_actual_accept": training_gate,
        "validation10_scene_pilot_has_actual_accept": pilot_gate,
        "complete_zero_prediction_sha256_identity": (
            set(hard_bypass) == set(S3_HARD_BYPASS_CONDITIONS) if is_s3 else True
        ),
        "query_hash_acceptance_engineering_failures_zero": True,
    }
    authorized = all(checks.values())
    manifest = {
        "schema": (
            "visfuse3d_stage019_s3b_fullval_gate_v1"
            if is_s3
            else "visfuse3d_stage019_s2b_fullval_gate_v1"
        ),
        "protocol_profile": args.protocol_profile,
        "status": (
            (
                "authorized_s3b_five_condition_full_validation"
                if is_s3
                else "authorized_s2b_seven_condition_full_validation"
            )
            if authorized
            else "stopped_gate_failed_negative_result_preserved"
        ),
        "authorized": authorized,
        "checks": checks,
        "clean_main_delta": main_delta["clean"],
        "fault_macro_delta": fault_macro,
        "jointly_improved_fault_conditions": jointly_improved,
        "training20_accepted_count": int(training.get("accepted_count", 0)),
        "pilot_accepted_count": sum(
            int(payload.get("accepted_count", 0)) for payload in pilot.values()
        ),
        "candidate_budget": budget,
        "inputs": {
            "margins": {"path": str(args.margins.resolve()), "sha256": sha256_file(args.margins)},
            "training_audit": {
                "path": str(args.training_audit_manifest.resolve()),
                "sha256": sha256_file(args.training_audit_manifest),
            },
            "s2a": [
                {"path": str(path.resolve()), "sha256": sha256_file(path)}
                for path in args.s2a_manifest
            ],
            "pilot": [
                {"path": str(path.resolve()), "sha256": sha256_file(path)}
                for path in args.pilot_manifest
            ],
            "hard_bypass": [
                {"path": str(path.resolve()), "sha256": sha256_file(path)}
                for path in args.hard_bypass_manifest
            ],
        },
        "claim_boundary": "execution_authorization_only_not_scientific_result",
    }
    atomic_write_json(output_dir / "s2b_fullval_gate_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0 if authorized else 3


if __name__ == "__main__":
    raise SystemExit(main())
