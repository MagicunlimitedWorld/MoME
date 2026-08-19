"""Calibration selection and pilot/full scientific gates for Stage019-S4."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .common import atomic_write_json
    from .extract_route_loss_cache import is_relative_to
    from .stage019_s4_contracts import (
        ACTIONABLE_CONDITIONS,
        FAULT_CONDITIONS,
        FUSION_STRENGTH_GRID,
        HARD_BYPASS_CONDITIONS,
        PROTOCOL_PROFILE,
        choose_g1_strength,
        evaluate_validation_gate,
        sha256_file,
    )
except ImportError:
    from common import atomic_write_json
    from extract_route_loss_cache import is_relative_to
    from stage019_s4_contracts import (
        ACTIONABLE_CONDITIONS,
        FAULT_CONDITIONS,
        FUSION_STRENGTH_GRID,
        HARD_BYPASS_CONDITIONS,
        PROTOCOL_PROFILE,
        choose_g1_strength,
        evaluate_validation_gate,
        sha256_file,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("gace_lite", "cp_afr"), required=True)
    parser.add_argument("--phase", choices=("calibration", "pilot", "fullval"), required=True)
    parser.add_argument("--seed-role", choices=("primary", "confirmation"), default="primary")
    parser.add_argument("--condition-manifest", type=Path, action="append", required=True)
    parser.add_argument("--temperature-condition-manifest", type=Path, action="append", default=[])
    parser.add_argument("--hard-bypass-manifest", type=Path, action="append", required=True)
    parser.add_argument("--selected-strength", type=float)
    parser.add_argument("--g1-gate", type=Path)
    parser.add_argument("--calibration-gate", type=Path)
    parser.add_argument("--pilot-gate", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    return parser.parse_args()


def _load_condition(path: Path, phase: str) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    condition = str(payload.get("condition"))
    metrics = payload.get("metrics", {})
    if (
        payload.get("schema") != "visfuse3d_stage019_s4_condition_merge_manifest_v1"
        or payload.get("status") != "complete_condition_evaluation"
        or payload.get("protocol_profile") != PROTOCOL_PROFILE
        or payload.get("phase") != phase
        or condition not in ACTIONABLE_CONDITIONS
        or int(payload.get("engineering_failure_count", -1)) != 0
        or not {"original_mome", "final"} <= set(metrics)
    ):
        raise ValueError(f"invalid S4 {phase} condition manifest: {path}")
    expected = {"pilot": (10, 401), "fullval": (150, 6019)}.get(phase)
    expected_split = "train" if phase == "calibration" else "val"
    if payload.get("source_split") != expected_split:
        raise ValueError(f"S4 {phase} source split must be {expected_split}")
    if expected is not None and (
        int(payload.get("scene_count", -1)), int(payload.get("frame_count", -1))
    ) != expected:
        raise ValueError(f"S4 {phase} coverage differs from preregistration")
    if phase == "calibration" and int(payload.get("scene_count", -1)) != 20:
        raise ValueError("S4 calibration requires exactly 20 train scenes")
    if int(payload.get("unique_token_count", -1)) != int(payload.get("frame_count", -2)):
        raise ValueError("S4 condition manifest contains duplicate frame tokens")
    return payload


def _delta(payload: dict, left: str = "final", right: str = "original_mome") -> dict[str, float]:
    return {
        metric: float(payload["metrics"][left][metric])
        - float(payload["metrics"][right][metric])
        for metric in ("mAP", "NDS")
    }


def _group_grid(paths: list[Path], phase: str) -> dict[float, dict[str, dict]]:
    output = {}
    for path in paths:
        payload = _load_condition(path, phase)
        identity = payload.get("locked_eval_identity") or {}
        strength = float(identity.get("fusion_strength", -1.0))
        condition = str(payload["condition"])
        output.setdefault(strength, {})
        if condition in output[strength]:
            raise ValueError("duplicate S4 strength/condition manifest")
        output[strength][condition] = payload
    if set(output) != set(FUSION_STRENGTH_GRID):
        raise ValueError("G1 calibration requires exact five-value strength grid")
    if any(set(values) != set(ACTIONABLE_CONDITIONS) for values in output.values()):
        raise ValueError("G1 strength grid lacks an actionable condition")
    locked = set()
    for conditions in output.values():
        for payload in conditions.values():
            identity = payload["locked_eval_identity"]
            checkpoint = identity.get("head_checkpoint") or {}
            locked.add(
                (
                    identity.get("mode"),
                    checkpoint.get("sha256"),
                    json.dumps(identity.get("source_hashes"), sort_keys=True),
                )
            )
    if len(locked) != 1:
        raise ValueError("G1 grid mixes checkpoint/mode/source identity")
    return output


def _load_single_set(paths: list[Path], phase: str) -> dict[str, dict]:
    if len(paths) != len(ACTIONABLE_CONDITIONS):
        raise ValueError("S4 gate requires exactly five actionable condition manifests")
    output = {}
    for path in paths:
        payload = _load_condition(path, phase)
        condition = str(payload["condition"])
        if condition in output:
            raise ValueError("duplicate S4 condition manifest")
        output[condition] = payload
    if set(output) != set(ACTIONABLE_CONDITIONS):
        raise ValueError("S4 actionable condition set is incomplete")
    identities = {
        json.dumps(payload.get("locked_eval_identity"), sort_keys=True)
        for payload in output.values()
    }
    if len(identities) != 1:
        raise ValueError("S4 condition set mixes eval identity")
    return output


def _load_zero(paths: list[Path], phase: str) -> dict[str, dict]:
    if len(paths) != len(HARD_BYPASS_CONDITIONS):
        raise ValueError("S4 gate requires two complete-zero manifests")
    output = {}
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        condition = str(payload.get("condition"))
        evidence = payload.get("hard_bypass_identity")
        expected_split = "train" if phase == "calibration" else "val"
        expected = {"pilot": (10, 401), "fullval": (150, 6019)}.get(phase)
        if (
            payload.get("schema") != "visfuse3d_stage019_s4_condition_merge_manifest_v1"
            or payload.get("status") != "complete_condition_evaluation"
            or payload.get("protocol_profile") != PROTOCOL_PROFILE
            or payload.get("phase") != phase
            or condition not in HARD_BYPASS_CONDITIONS
            or condition in output
            or not isinstance(evidence, dict)
            or int(payload.get("engineering_failure_count", -1)) != 0
            or payload.get("source_split") != expected_split
            or (phase == "calibration" and int(payload.get("scene_count", -1)) != 20)
            or (
                expected is not None
                and (
                    int(payload.get("scene_count", -1)),
                    int(payload.get("frame_count", -1)),
                )
                != expected
            )
            or int(payload.get("unique_token_count", -1))
            != int(payload.get("frame_count", -2))
        ):
            raise ValueError(f"invalid S4 hard-bypass manifest: {path}")
        output[condition] = payload
    return output


def _checkpoint_seed(payload: dict) -> int:
    identity = payload.get("locked_eval_identity") or {}
    checkpoint = identity.get("head_checkpoint") or {}
    # The checkpoint schema/seed was already fail-closed by eval_worker; merge
    # retains its path and hash, so read and verify only metadata here.
    path = Path(str(checkpoint.get("path", "")))
    if not path.is_file() or sha256_file(path) != checkpoint.get("sha256"):
        raise ValueError("S4 gate head checkpoint path/hash changed")
    import torch

    raw = torch.load(str(path), map_location="cpu")
    return int(raw.get("seed", -1))


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    artifact_root = args.artifact_root.resolve()
    source_root = Path(__file__).resolve().parents[2]
    if not is_relative_to(output_dir, artifact_root):
        raise ValueError("S4 gate output must stay inside artifact root")
    if is_relative_to(output_dir, source_root):
        raise ValueError("S4 gate output cannot enter source checkout")
    if output_dir.exists():
        raise FileExistsError(f"immutable S4 gate exists: {output_dir}")
    zero_payloads = _load_zero(args.hard_bypass_manifest, args.phase)
    selection = None
    temperature_report = None
    if args.method == "gace_lite" and args.phase == "calibration":
        grid = _group_grid(args.condition_manifest, args.phase)
        selection = choose_g1_strength(
            {
                strength: {
                    condition: _delta(payload)
                    for condition, payload in conditions.items()
                }
                for strength, conditions in grid.items()
            }
        )
        if selection["selected_strength"] is None:
            selected = grid[0.0]
        else:
            selected = grid[float(selection["selected_strength"])]
        if len(args.temperature_condition_manifest) != 25:
            raise ValueError("G1 calibration must report the temperature-only 5x5 grid")
        temperature_grid = _group_grid(args.temperature_condition_manifest, args.phase)
        temperature_report = choose_g1_strength(
            {
                strength: {
                    condition: _delta(payload)
                    for condition, payload in conditions.items()
                }
                for strength, conditions in temperature_grid.items()
            }
        )
    else:
        if args.temperature_condition_manifest:
            raise ValueError("temperature-only baseline is calibration reporting only")
        selected = _load_single_set(args.condition_manifest, args.phase)
        if args.method == "gace_lite":
            if args.selected_strength is None or float(args.selected_strength) not in FUSION_STRENGTH_GRID:
                raise ValueError("pilot/full GACE gate requires locked selected strength")
            observed = {
                float((payload.get("locked_eval_identity") or {}).get("fusion_strength", -1.0))
                for payload in selected.values()
            }
            if observed != {float(args.selected_strength)}:
                raise ValueError("GACE manifest strength differs from calibration lock")
        else:
            observed_modes = {
                str((payload.get("locked_eval_identity") or {}).get("mode"))
                for payload in selected.values()
            }
            if observed_modes == {"stacked"}:
                if args.g1_gate is None or args.selected_strength is None:
                    raise ValueError("stacked CP-AFR requires the locked primary G1 gate/strength")
                g1_gate = json.loads(args.g1_gate.read_text(encoding="utf-8"))
                locked_strength = (
                    g1_gate.get("g1_strength_selection", {}).get("selected_strength")
                )
                if (
                    g1_gate.get("schema")
                    != "visfuse3d_stage019_s4_validation_gate_manifest_v1"
                    or g1_gate.get("phase") != "calibration"
                    or g1_gate.get("seed_role") != "primary"
                    or not bool(g1_gate.get("passed"))
                    or locked_strength is None
                    or float(locked_strength) != float(args.selected_strength)
                ):
                    raise ValueError("stacked CP-AFR G1 calibration authority mismatch")
                observed_strengths = {
                    float((payload.get("locked_eval_identity") or {}).get("fusion_strength", -1.0))
                    for payload in selected.values()
                }
                if observed_strengths != {float(locked_strength)}:
                    raise ValueError("stacked CP-AFR used a different G1 strength")
                cp_identity = next(iter(selected.values()))["locked_eval_identity"]
                cp_path = Path(cp_identity["head_checkpoint"]["path"])
                import torch

                cp_payload = torch.load(str(cp_path), map_location="cpu")
                parent_gace = cp_payload.get("parent_gace_checkpoint") or {}
                locked_gace = (
                    g1_gate.get("candidate_eval_identity", {}).get("head_checkpoint", {})
                )
                if (
                    not parent_gace
                    or parent_gace.get("sha256") != locked_gace.get("sha256")
                ):
                    raise ValueError("stacked CP-AFR parent differs from passed G1 checkpoint")
            elif observed_modes == {"attribute_only"}:
                if args.g1_gate is not None or args.selected_strength is not None:
                    raise ValueError("attribute-only CP-AFR must stay on raw scores")
                observed_strengths = {
                    float((payload.get("locked_eval_identity") or {}).get("fusion_strength", -1.0))
                    for payload in selected.values()
                }
                if observed_strengths != {1.0}:
                    raise ValueError("attribute-only CP-AFR strength identity drifted")
            else:
                raise ValueError("CP-AFR condition manifests mix fusion modes")
    scene_identities = {
        tuple(str(value) for value in payload.get("scene_tokens", []))
        for payload in selected.values()
    } | {
        tuple(str(value) for value in payload.get("scene_tokens", []))
        for payload in zero_payloads.values()
    }
    if len(scene_identities) != 1:
        raise ValueError("S4 actionable/zero scene identities differ")
    frame_identities = {
        str(payload.get("frame_token_order_sha256"))
        for payload in selected.values()
    } | {
        str(payload.get("frame_token_order_sha256"))
        for payload in zero_payloads.values()
    }
    if len(frame_identities) != 1 or "None" in frame_identities:
        raise ValueError("S4 actionable/zero frame token identities differ")
    zero = {
        condition: payload["hard_bypass_identity"]
        for condition, payload in zero_payloads.items()
    }
    authority_records = {}
    if args.phase in ("pilot", "fullval"):
        if args.calibration_gate is None:
            raise ValueError("pilot/full gate requires passed calibration authority")
        calibration_gate = json.loads(args.calibration_gate.read_text(encoding="utf-8"))
        if (
            calibration_gate.get("schema")
            != "visfuse3d_stage019_s4_validation_gate_manifest_v1"
            or calibration_gate.get("phase") != "calibration"
            or calibration_gate.get("seed_role") != "primary"
            or calibration_gate.get("method") != args.method
            or not bool(calibration_gate.get("passed"))
        ):
            raise ValueError("S4 calibration authority is not a passed primary gate")
        current_identity = next(iter(selected.values())).get("locked_eval_identity", {})
        calibration_identity = calibration_gate.get("candidate_eval_identity", {})
        if (
            (current_identity.get("head_checkpoint") or {}).get("sha256")
            != (calibration_identity.get("head_checkpoint") or {}).get("sha256")
            or current_identity.get("mode") != calibration_identity.get("mode")
            or current_identity.get("source_hashes")
            != calibration_identity.get("source_hashes")
        ):
            raise ValueError("S4 pilot/full candidate differs from calibration authority")
        if args.method == "gace_lite":
            authority_strength = calibration_gate.get("g1_strength_selection", {}).get(
                "selected_strength"
            )
            if (
                authority_strength is None
                or float(authority_strength) != float(args.selected_strength)
            ):
                raise ValueError("GACE pilot/full strength differs from calibration authority")
        authority_records["calibration_gate"] = {
            "path": str(args.calibration_gate.resolve()),
            "sha256": sha256_file(args.calibration_gate),
        }
    elif args.calibration_gate is not None:
        raise ValueError("calibration phase cannot cite itself as authority")
    if args.phase == "fullval":
        if args.pilot_gate is None:
            raise ValueError("fullval gate requires passed pilot authority")
        pilot_gate = json.loads(args.pilot_gate.read_text(encoding="utf-8"))
        if (
            pilot_gate.get("schema")
            != "visfuse3d_stage019_s4_validation_gate_manifest_v1"
            or pilot_gate.get("phase") != "pilot"
            or pilot_gate.get("seed_role") != "primary"
            or pilot_gate.get("method") != args.method
            or not bool(pilot_gate.get("passed"))
        ):
            raise ValueError("S4 pilot authority is not a passed primary gate")
        pilot_identity = pilot_gate.get("candidate_eval_identity", {})
        current_identity = next(iter(selected.values())).get("locked_eval_identity", {})
        if (
            (pilot_identity.get("head_checkpoint") or {}).get("sha256")
            != (current_identity.get("head_checkpoint") or {}).get("sha256")
            or pilot_identity.get("mode") != current_identity.get("mode")
            or pilot_identity.get("fusion_strength")
            != current_identity.get("fusion_strength")
        ):
            raise ValueError("S4 fullval candidate differs from pilot authority")
        authority_records["pilot_gate"] = {
            "path": str(args.pilot_gate.resolve()),
            "sha256": sha256_file(args.pilot_gate),
        }
    elif args.pilot_gate is not None:
        raise ValueError("only fullval may cite a pilot gate")
    seeds = {_checkpoint_seed(payload) for payload in selected.values()}
    expected_seed = 20260710 if args.seed_role == "primary" else 20260711
    if seeds != {expected_seed}:
        raise ValueError("S4 condition checkpoints differ from the requested seed role")
    deltas = {condition: _delta(payload) for condition, payload in selected.items()}
    changed = sum(int(payload.get("changed_object_count", 0)) for payload in selected.values())
    accepted = sum(int(payload.get("accepted_object_count", 0)) for payload in selected.values())
    rescued = sum(int(payload.get("rescued_object_count", 0)) for payload in selected.values())
    attribute_vs_score = None
    if args.method == "cp_afr":
        if any("score_baseline" not in payload.get("metrics", {}) for payload in selected.values()):
            raise ValueError("CP-AFR manifests lack score baseline metrics")
        attribute_vs_score = {
            condition: float(payload["metrics"]["final"]["NDS"])
            - float(payload["metrics"]["score_baseline"]["NDS"])
            for condition, payload in selected.items()
            if condition in FAULT_CONDITIONS
        }
    decision = evaluate_validation_gate(
        deltas,
        changed,
        accepted,
        zero,
        args.method,
        attribute_vs_score,
    )
    if selection is not None:
        decision["g1_strength_selection"] = selection
        decision["temperature_scaling_baseline"] = temperature_report
        decision["checks"]["selected_g1_strength_passes_full_calibration_gate"] = bool(
            decision["passed"] and selection["passed"]
        )
        decision["passed"] = all(decision["checks"].values())
        decision["status"] = (
            "passed" if decision["passed"] else "stopped_scientific_gate_failed"
        )
    if args.seed_role == "confirmation":
        fault_map = sum(deltas[condition]["mAP"] for condition in FAULT_CONDITIONS) / 4.0
        fault_nds = sum(deltas[condition]["NDS"] for condition in FAULT_CONDITIONS) / 4.0
        direction = bool(fault_map > 0.0 and fault_nds >= 0.0)
        full_safety_diagnostic = bool(decision["passed"])
        evidence_integrity = bool(
            decision["checks"]["complete_zero_prediction_sha256_identity"]
        )
        decision["confirmation_direction_passed"] = direction
        decision["confirmation_full_formal_gate_diagnostic"] = full_safety_diagnostic
        decision["confirmation_evidence_integrity_passed"] = evidence_integrity
        decision["passed"] = bool(direction and evidence_integrity)
        decision["status"] = (
            "passed_confirmation_direction_only_not_primary_substitute"
            if decision["passed"]
            else "stopped_confirmation_direction_failed"
        )
    manifest = {
        **decision,
        "schema": "visfuse3d_stage019_s4_validation_gate_manifest_v1",
        "protocol_profile": PROTOCOL_PROFILE,
        "phase": args.phase,
        "seed_role": args.seed_role,
        "candidate_eval_identity": next(iter(selected.values())).get(
            "locked_eval_identity"
        ),
        "rescued_object_count": rescued,
        "condition_inputs": [
            {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for path in args.condition_manifest
        ],
        "temperature_condition_inputs": [
            {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for path in args.temperature_condition_manifest
        ],
        "hard_bypass_inputs": [
            {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for path in args.hard_bypass_manifest
        ],
        "g1_gate": (
            None
            if args.g1_gate is None
            else {"path": str(args.g1_gate.resolve()), "sha256": sha256_file(args.g1_gate)}
        ),
        "phase_authorities": authority_records,
        "claim_boundary": (
            "calibration_selection_only" if args.phase == "calibration" else (
                "pilot_go_no_go_only" if args.phase == "pilot" else "full_validation_gate"
            )
        ),
    }
    output_dir.mkdir(parents=True)
    atomic_write_json(output_dir / "stage019_s4_validation_gate_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0 if bool(manifest["passed"]) else 3


if __name__ == "__main__":
    raise SystemExit(main())
