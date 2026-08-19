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
except ImportError:
    from common import atomic_write_json
    from extract_route_loss_cache import is_relative_to, sha256_file
    from full_query_oracle_worker import CONDITIONS


FAULT_CONDITIONS = tuple(value for value in CONDITIONS if value != "clean")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--margins", type=Path, required=True)
    parser.add_argument("--training-audit-manifest", type=Path, required=True)
    parser.add_argument("--s2a-manifest", type=Path, action="append", required=True)
    parser.add_argument("--pilot-manifest", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    return parser.parse_args()


def _load_by_condition(paths: list[Path], mode: str, official: bool) -> dict[str, dict]:
    if len(paths) != len(CONDITIONS):
        raise ValueError("gate requires exactly one manifest for each of seven conditions")
    output = {}
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        condition = str(payload.get("condition"))
        required_status = "complete_official_evaluation" if official else "complete_diagnostic_merge"
        if (
            payload.get("status") != required_status
            or payload.get("mode") != mode
            or condition not in CONDITIONS
            or int(payload.get("engineering_failure_count", -1)) != 0
            or condition in output
        ):
            raise ValueError(f"invalid/duplicate gate manifest: {path}")
        output[condition] = payload
    if set(output) != set(CONDITIONS):
        raise ValueError("gate condition set is incomplete")
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
    budget = margins.get("candidate_budget", {})
    calibration_gate = bool(
        margins.get("status") == "passed"
        and budget.get("status") == "coverage_reached"
        and int(budget.get("K", -1)) in (4, 8, 16)
        and float(budget.get("coverage", {}).get(str(budget.get("K")), 0.0)) >= 0.95
    )
    training = json.loads(args.training_audit_manifest.read_text(encoding="utf-8"))
    training_gate = bool(
        training.get("status") == "passed_locked_greedy_audit"
        and bool(training.get("locked_greedy_audit"))
        and int(training.get("frame_condition_count", -1)) == 5621
        and int(training.get("accepted_count", 0)) > 0
        and training.get("locked_margins_sha256") == sha256_file(args.margins)
    )
    s2a = _load_by_condition(args.s2a_manifest, mode="s2a", official=True)
    pilot = _load_by_condition(args.pilot_manifest, mode="s2b", official=False)
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
            np.mean([float(main_delta[condition][metric]) for condition in FAULT_CONDITIONS])
        )
        for metric in ("mAP", "NDS")
    }
    macro_gate = fault_macro["mAP"] > 0.0 and fault_macro["NDS"] > 0.0
    jointly_improved = [
        condition
        for condition in FAULT_CONDITIONS
        if float(main_delta[condition]["mAP"]) > 0.0
        and float(main_delta[condition]["NDS"]) > 0.0
    ]
    same_condition_gate = bool(jointly_improved)
    checks = {
        "s2a_clean_map_nds_non_decreasing": clean_gate,
        "six_fault_macro_map_nds_strictly_positive": macro_gate,
        "same_fault_condition_map_nds_strictly_positive": same_condition_gate,
        "calibration_K_nonzero_and_95pct_coverage": calibration_gate,
        "training20_has_locked_actual_accept": training_gate,
        "validation10_scene_pilot_has_actual_accept": pilot_gate,
        "query_hash_acceptance_engineering_failures_zero": True,
    }
    authorized = all(checks.values())
    manifest = {
        "schema": "visfuse3d_stage019_s2b_fullval_gate_v1",
        "status": (
            "authorized_s2b_seven_condition_full_validation"
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
        },
        "claim_boundary": "execution_authorization_only_not_scientific_result",
    }
    atomic_write_json(output_dir / "s2b_fullval_gate_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0 if authorized else 3


if __name__ == "__main__":
    raise SystemExit(main())
