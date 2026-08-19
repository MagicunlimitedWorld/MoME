"""Aggregate a locked Stage019-S2 repeat profile and lock margins/K."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

try:
    from .common import atomic_write_json
    from .extract_route_loss_cache import is_relative_to, sha256_file
    from .stage019_s2_training_conditions import (
        CONDITIONS,
        S3_ACTIONABLE_CONDITIONS,
        S3_HARD_BYPASS_CONDITIONS,
    )
except ImportError:
    from common import atomic_write_json
    from extract_route_loss_cache import is_relative_to, sha256_file
    from stage019_s2_training_conditions import (
        CONDITIONS,
        S3_ACTIONABLE_CONDITIONS,
        S3_HARD_BYPASS_CONDITIONS,
    )


REPEAT_PROFILES = {
    "four_repeat_v1": (
        "gpu0_repeat0",
        "gpu1_repeat0",
        "gpu0_repeat1",
        "gpu1_repeat1",
    ),
    "three_repeat_v1": (
        "gpu0_repeat0",
        "gpu0_repeat1",
        "gpu1_repeat0",
    ),
}
CANONICAL_REPEAT_ID = "gpu0_repeat0"
EXPECTED_FRAMES = 803
EXPECTED_QUERIES = 900
CANDIDATE_KS = (4, 8, 16)
REQUIRED_COVERAGE = 0.95
LEGACY_PROTOCOL_PROFILE = "stage019_s2_legacy_v1"
S3_PROTOCOL_PROFILE = "stage019_s3_actionable_hard_bypass_v1"
PROTOCOL_PROFILES = (LEGACY_PROTOCOL_PROFILE, S3_PROTOCOL_PROFILE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeat-dir", type=Path, action="append", required=True)
    parser.add_argument(
        "--repeat-profile",
        choices=tuple(REPEAT_PROFILES),
        default="four_repeat_v1",
    )
    parser.add_argument("--canonical-repeat-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument(
        "--protocol-profile",
        choices=PROTOCOL_PROFILES,
        default=LEGACY_PROTOCOL_PROFILE,
    )
    parser.add_argument("--source-calibration-root", type=Path)
    parser.add_argument(
        "--expected-repeat-manifest-sha256",
        action="append",
        default=[],
        metavar="REPEAT_ID=SHA256",
    )
    parser.add_argument("--allow-smoke", action="store_true")
    return parser.parse_args()


def protocol_conditions(profile: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if profile == S3_PROTOCOL_PROFILE:
        return S3_ACTIONABLE_CONDITIONS, S3_HARD_BYPASS_CONDITIONS
    return CONDITIONS, ()


def query_margin_conditions(profile: str) -> tuple[str, ...]:
    """Return conditions needing S2-A numerical envelopes.

    S3 narrows only the S2-B proxy audit and search.  S2-A still reports all
    seven conditions, so its keep/reconstruction envelopes must be derived
    again from the locked raw repeats rather than copied from the failed S2
    aggregate.
    """

    if profile not in PROTOCOL_PROFILES:
        raise ValueError(f"unknown protocol profile: {profile}")
    return CONDITIONS


def expected_manifest_hashes(values: list[str]) -> dict[str, str]:
    output = {}
    for value in values:
        repeat_id, separator, digest = value.partition("=")
        if not separator or not repeat_id or len(digest) != 64:
            raise ValueError(
                "expected repeat manifest hashes must use REPEAT_ID=64_HEX_SHA256"
            )
        try:
            int(digest, 16)
        except ValueError as exc:
            raise ValueError("repeat manifest SHA256 is not hexadecimal") from exc
        if repeat_id in output:
            raise ValueError(f"duplicate expected repeat manifest hash: {repeat_id}")
        output[repeat_id] = digest.upper()
    return output


def expected_repeat_ids(profile: str) -> tuple[str, ...]:
    try:
        return REPEAT_PROFILES[profile]
    except KeyError as exc:
        raise ValueError(f"unknown calibration repeat profile: {profile}") from exc


def claim_boundary(profile: str) -> str:
    count_word = "three" if profile == "three_repeat_v1" else "four"
    return (
        f"observed_{count_word}_repeat_numerical_envelope_"
        "not_confidence_interval_or_absolute_bound"
    )


def validate_repeat_identity(
    profile: str, repeat_ids: list[str], canonical_repeat_id: str
) -> None:
    expected = expected_repeat_ids(profile)
    if canonical_repeat_id != CANONICAL_REPEAT_ID:
        raise ValueError(
            f"{profile} canonical repeat must be {CANONICAL_REPEAT_ID}"
        )
    if len(repeat_ids) != len(expected):
        raise ValueError(
            f"{profile} requires exactly {len(expected)} calibration repeats"
        )
    if len(repeat_ids) != len(set(repeat_ids)):
        raise ValueError("calibration repeat ids must be unique")
    if set(repeat_ids) != set(expected):
        raise ValueError(
            f"{profile} repeat ids must be exactly {list(expected)}"
        )


def observed_numerical_envelope(
    values: np.ndarray, axis: int | None = None
) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("numerical envelope requires finite observations")
    return np.ptp(array, axis=axis)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {key: payload[key] for key in payload.files}


def _require_finite(name: str, value: np.ndarray) -> None:
    if not np.isfinite(value).all():
        raise ValueError(f"non-finite calibration array: {name}")


def _validate_array_bundle(bundle: dict[str, np.ndarray], label: str) -> None:
    shapes = {
        "base_fixed_query_losses": (EXPECTED_QUERIES,),
        "route_fixed_query_losses": (EXPECTED_QUERIES, 3),
        "reconstruction_fixed_query_losses": (EXPECTED_QUERIES,),
        "full_context_gains": (EXPECTED_QUERIES, 3),
        "base_routes": (EXPECTED_QUERIES,),
        "candidate_quality": (EXPECTED_QUERIES, 3),
    }
    if set(bundle) != set(shapes):
        raise ValueError(f"{label} calibration arrays have unexpected keys")
    for key, shape in shapes.items():
        if bundle[key].shape != shape:
            raise ValueError(f"{label}:{key} expected {shape}, got {bundle[key].shape}")
        _require_finite(f"{label}:{key}", bundle[key])
    routes = bundle["base_routes"]
    if not np.isin(routes, np.asarray([0, 1, 2], dtype=routes.dtype)).all():
        raise ValueError(f"{label}: base route outside {{0,1,2}}")
    reconstructed = np.take_along_axis(
        bundle["route_fixed_query_losses"], routes[:, None].astype(np.int64), axis=1
    )[:, 0]
    if not np.array_equal(reconstructed, bundle["reconstruction_fixed_query_losses"]):
        raise ValueError(f"{label}: reconstruction losses do not match base routes")
    gains = reconstructed[:, None] - bundle["route_fixed_query_losses"]
    if not np.array_equal(gains, bundle["full_context_gains"]):
        raise ValueError(f"{label}: stored full-context gains are inconsistent")


def _candidate_key(item: dict) -> tuple[int, int]:
    return int(item["query_id"]), int(item["destination_expert"])


def _trace_by_step(payload: dict, label: str) -> list[dict]:
    trace = payload.get("trace")
    if not isinstance(trace, list) or len(trace) > 16:
        raise ValueError(f"{label}: invalid calibration trace length")
    if [int(row.get("step", -1)) for row in trace] != list(range(len(trace))):
        raise ValueError(f"{label}: trace steps are not contiguous")
    keys = [_candidate_key(row) for row in trace]
    if len(keys) != len(set(keys)):
        raise ValueError(f"{label}: duplicate query/destination action")
    candidate_order = [_candidate_key(row) for row in payload.get("candidate_order", [])]
    if keys != candidate_order[: len(keys)]:
        raise ValueError(f"{label}: trace is not the locked candidate prefix")
    return trace


def _locked_source_identity(manifest: dict) -> dict:
    keys = (
        "checkpoint_sha256",
        "config_sha256",
        "training20_manifest_sha256",
        "worker_sha256",
        "detector_sha256",
        "med_sha256",
        "router_sha256",
        "trainable_parameter_count",
        "conditions",
        "candidate_limit",
    )
    return {key: manifest.get(key) for key in keys}


def _budget_from_quality(total: float, captured: dict[int, float]) -> dict:
    if total == 0.0:
        return {
            "status": "zero_positive_quality",
            "K": 0,
            "total_positive_quality": 0.0,
            "coverage": {},
        }
    coverage = {str(key): float(captured[key] / total) for key in CANDIDATE_KS}
    for key in CANDIDATE_KS:
        if coverage[str(key)] >= REQUIRED_COVERAGE:
            return {
                "status": "coverage_reached",
                "K": key,
                "total_positive_quality": float(total),
                "coverage": coverage,
            }
    return {
        "status": "candidate_mass_not_coverable_under_K16",
        "K": None,
        "total_positive_quality": float(total),
        "coverage": coverage,
    }


def main() -> int:
    args = parse_args()
    expected_ids = expected_repeat_ids(args.repeat_profile)
    actionable_conditions, hard_bypass_conditions = protocol_conditions(
        args.protocol_profile
    )
    s2a_conditions = query_margin_conditions(args.protocol_profile)
    expected_hashes = expected_manifest_hashes(
        args.expected_repeat_manifest_sha256
    )
    source_calibration_root = (
        args.source_calibration_root.resolve()
        if args.source_calibration_root is not None
        else None
    )
    if args.protocol_profile == S3_PROTOCOL_PROFILE:
        if source_calibration_root is None:
            raise ValueError("S3 reaggregation requires --source-calibration-root")
        if set(expected_hashes) != set(expected_ids):
            raise ValueError("S3 reaggregation requires one locked hash per repeat manifest")
    elif args.source_calibration_root is not None or expected_hashes:
        raise ValueError("source calibration locks are reserved for the S3 profile")
    if len(args.repeat_dir) != len(expected_ids):
        raise ValueError(
            f"{args.repeat_profile} requires exactly {len(expected_ids)} "
            "calibration repeat directories"
        )
    output_dir = args.output_dir.resolve()
    artifact_root = args.artifact_root.resolve()
    source_root = Path(__file__).resolve().parents[2]
    if not is_relative_to(output_dir, artifact_root):
        raise ValueError("calibration aggregate must stay below the S2 artifact root")
    if is_relative_to(output_dir, source_root):
        raise ValueError("calibration aggregate cannot enter the source checkout")
    if output_dir.exists():
        raise FileExistsError(f"immutable calibration aggregate exists: {output_dir}")
    output_dir.mkdir(parents=True)

    manifests = []
    repeat_by_id = {}
    allowed_statuses = {"passed", "smoke_passed"} if args.allow_smoke else {"passed"}
    for repeat_dir in args.repeat_dir:
        repeat_dir = repeat_dir.resolve()
        if source_calibration_root is not None and not is_relative_to(
            repeat_dir, source_calibration_root
        ):
            raise ValueError("S3 repeat directory escaped the locked source root")
        manifest_path = repeat_dir / "calibration_repeat_manifest.json"
        manifest = _load_json(manifest_path)
        repeat_id = str(manifest.get("repeat_id"))
        if manifest.get("status") not in allowed_statuses:
            raise ValueError(f"repeat is not complete: {repeat_dir}")
        if repeat_id in repeat_by_id:
            raise ValueError(f"duplicate repeat id: {repeat_id}")
        if expected_hashes and sha256_file(manifest_path).upper() != expected_hashes.get(
            repeat_id
        ):
            raise ValueError(f"locked source repeat manifest hash drifted: {repeat_id}")
        repeat_by_id[repeat_id] = repeat_dir
        manifests.append((repeat_dir, manifest_path, manifest))
    validate_repeat_identity(
        args.repeat_profile,
        [str(item[2].get("repeat_id")) for item in manifests],
        args.canonical_repeat_id,
    )
    if args.canonical_repeat_id not in repeat_by_id:
        raise ValueError("canonical repeat id is absent")
    canonical_dir = repeat_by_id[args.canonical_repeat_id]
    canonical_manifest = next(
        manifest for directory, _, manifest in manifests if directory == canonical_dir
    )
    if bool(canonical_manifest.get("canonical_replay")):
        raise ValueError("canonical repeat cannot itself be a replay")
    if sum(not bool(item[2].get("canonical_replay")) for item in manifests) != 1:
        raise ValueError("exactly one repeat must define the canonical trajectory")
    locked_identity = _locked_source_identity(canonical_manifest)
    for _, _, manifest in manifests:
        if _locked_source_identity(manifest) != locked_identity:
            raise ValueError("calibration repeats have different locked source identity")
        if tuple(manifest.get("conditions", [])) != tuple(CONDITIONS):
            raise ValueError("calibration repeat does not cover the locked seven conditions")
        expected_count = len(CONDITIONS) * EXPECTED_FRAMES
        if not args.allow_smoke and int(manifest.get("frame_condition_count", -1)) != expected_count:
            raise ValueError("calibration repeat does not cover 5,621 frame-conditions")

    epsilon_query = {
        condition: {
            "original_mome": np.zeros(3, dtype=np.float64),
            "reconstruction": np.zeros(3, dtype=np.float64),
        }
        for condition in s2a_conditions
    }
    proxy_overestimate = {
        condition: np.zeros(3, dtype=np.float64)
        for condition in actionable_conditions
    }
    proxy_audit_count = {
        condition: np.zeros(3, dtype=np.int64)
        for condition in actionable_conditions
    }
    epsilon_frame = {condition: 0.0 for condition in actionable_conditions}
    frame_action_audit_count = {
        condition: 0 for condition in actionable_conditions
    }
    low_margin_action_count = 0
    frame_count_by_condition = {}

    # First pass locks paired numerical envelopes and proxy over-estimation.
    for condition in s2a_conditions:
        frame_sets = []
        for repeat_dir, _, _ in manifests:
            frame_sets.append(
                {path.stem for path in (repeat_dir / condition / "frames").glob("*.json")}
            )
        if any(frame_set != frame_sets[0] for frame_set in frame_sets[1:]):
            raise ValueError(f"{condition}: repeat frame-token sets differ")
        frame_tokens = sorted(frame_sets[0])
        if not args.allow_smoke and len(frame_tokens) != EXPECTED_FRAMES:
            raise ValueError(f"{condition}: expected 803 paired frames")
        frame_count_by_condition[condition] = len(frame_tokens)
        for token in frame_tokens:
            arrays = []
            traces = []
            input_hashes = set()
            for repeat_dir, _, _ in manifests:
                label = f"{repeat_dir.name}/{condition}/{token}"
                trace_payload = _load_json(
                    repeat_dir / condition / "frames" / f"{token}.json"
                )
                bundle = _load_npz(
                    repeat_dir / condition / "frames" / f"{token}.npz"
                )
                _validate_array_bundle(bundle, label)
                arrays.append(bundle)
                traces.append(_trace_by_step(trace_payload, label))
                input_hashes.add(str(trace_payload.get("input_sha256")))
            if len(input_hashes) != 1:
                raise ValueError(f"{condition}/{token}: paired input hash drift")
            base_routes = arrays[0]["base_routes"]
            if any(not np.array_equal(item["base_routes"], base_routes) for item in arrays[1:]):
                raise ValueError(f"{condition}/{token}: paired base-route identity drift")
            keep_gains = np.stack(
                [item["base_fixed_query_losses"][:, None] - item["route_fixed_query_losses"] for item in arrays]
            ).astype(np.float64)
            reconstruction_gains = np.stack(
                [item["full_context_gains"] for item in arrays]
            ).astype(np.float64)
            for destination in range(3):
                epsilon_query[condition]["original_mome"][destination] = max(
                    epsilon_query[condition]["original_mome"][destination],
                    float(
                        observed_numerical_envelope(
                            keep_gains[:, :, destination], axis=0
                        ).max()
                    ),
                )
                epsilon_query[condition]["reconstruction"][destination] = max(
                    epsilon_query[condition]["reconstruction"][destination],
                    float(
                        observed_numerical_envelope(
                            reconstruction_gains[:, :, destination], axis=0
                        ).max()
                    ),
                )

            if condition not in actionable_conditions:
                continue

            canonical_index = next(
                index for index, item in enumerate(manifests)
                if item[0] == canonical_dir
            )
            canonical_trace = traces[canonical_index]
            canonical_keys = [_candidate_key(row) for row in canonical_trace]
            for trace in traces:
                if [_candidate_key(row) for row in trace] != canonical_keys:
                    raise ValueError(f"{condition}/{token}: replay candidate order drift")
            for step, canonical_row in enumerate(canonical_trace):
                rows = [trace[step] for trace in traces]
                for key in ("current_route_hash", "candidate_route_hash"):
                    if len({str(row.get(key)) for row in rows}) != 1:
                        raise ValueError(f"{condition}/{token}: replay {key} drift")
                destination = int(canonical_row["destination_expert"])
                query_id = int(canonical_row["query_id"])
                frame_gains = [row.get("actual_frame_gain") for row in rows]
                query_gains = [row.get("actual_query_gain") for row in rows]
                if all(value is None for value in frame_gains):
                    continue
                if any(value is None for value in frame_gains + query_gains):
                    raise ValueError(f"{condition}/{token}: partial paired action audit")
                observed_frame = np.asarray(frame_gains, dtype=np.float64)
                observed_query = np.asarray(query_gains, dtype=np.float64)
                if not np.isfinite(observed_frame).all() or not np.isfinite(observed_query).all():
                    raise ValueError(f"{condition}/{token}: non-finite paired action gain")
                epsilon_frame[condition] = max(
                    epsilon_frame[condition],
                    float(observed_numerical_envelope(observed_frame)),
                )
                frame_action_audit_count[condition] += 1
                proxy_audit_count[condition][destination] += 1
                full = reconstruction_gains[:, query_id, destination]
                proxy_overestimate[condition][destination] = max(
                    proxy_overestimate[condition][destination],
                    float(np.maximum(0.0, full - observed_query).max()),
                )
                low_margin_action_count += int(
                    np.count_nonzero(np.abs(observed_frame) <= 1e-4)
                )

    missing_proxy_cells = [
        f"{condition}:{destination}"
        for condition in actionable_conditions
        for destination in range(3)
        if int(proxy_audit_count[condition][destination]) == 0
    ]

    # Second pass applies the locked envelopes to the canonical repeat only.
    total_quality = 0.0
    captured_quality = {key: 0.0 for key in CANDIDATE_KS}
    positive_action_count = 0
    for condition in actionable_conditions:
        frame_dir = canonical_dir / condition / "frames"
        for npz_path in sorted(frame_dir.glob("*.npz")):
            bundle = _load_npz(npz_path)
            quality = (
                bundle["full_context_gains"].astype(np.float64)
                - epsilon_query[condition]["reconstruction"][None, :]
                - proxy_overestimate[condition][None, :]
            )
            quality = np.where(np.isfinite(quality) & (quality > 0), quality, 0.0)
            quality[
                np.arange(EXPECTED_QUERIES), bundle["base_routes"].astype(np.int64)
            ] = 0.0
            flattened = np.sort(quality.reshape(-1))[::-1]
            total_quality += float(flattened.sum(dtype=np.float64))
            positive_action_count += int(np.count_nonzero(flattened > 0))
            for key in CANDIDATE_KS:
                captured_quality[key] += float(flattened[:key].sum(dtype=np.float64))
    budget = _budget_from_quality(total_quality, captured_quality)
    status = "passed"
    if missing_proxy_cells:
        status = "calibration_incomplete"
    elif budget["status"] == "zero_positive_quality":
        status = "s2b_stopped_zero_positive_quality"
    elif budget["status"] != "coverage_reached":
        status = "candidate_mass_not_coverable_under_K16"

    calibration_claim_boundary = claim_boundary(args.repeat_profile)
    if args.protocol_profile == S3_PROTOCOL_PROFILE:
        calibration_claim_boundary += (
            "_derived_from_locked_training_only_raw_repeats_"
            "s2a_all_conditions_s2b_actionable_conditions_only"
        )
    applicability = {
        condition: {
            str(destination): "audited" for destination in range(3)
        }
        for condition in actionable_conditions
    }
    applicability.update(
        {
            condition: {
                str(destination): "hard_bypass_not_applicable"
                for destination in range(3)
            }
            for condition in hard_bypass_conditions
        }
    )

    margins = {
        "schema": "visfuse3d_stage019_s2_calibration_margins_v1",
        "status": status,
        "protocol_profile": args.protocol_profile,
        "repeat_profile": args.repeat_profile,
        "repeat_ids": list(expected_ids),
        "s2a_reporting_conditions": list(s2a_conditions),
        "actionable_conditions": list(actionable_conditions),
        "hard_bypass_conditions": list(hard_bypass_conditions),
        "condition_destination_applicability": applicability,
        "claim_boundary": calibration_claim_boundary,
        "epsilon_num_query": {
            condition: {
                baseline: values.tolist()
                for baseline, values in epsilon_query[condition].items()
            }
            for condition in s2a_conditions
        },
        "proxy_overestimate": {
            condition: proxy_overestimate[condition].tolist()
            for condition in actionable_conditions
        },
        "epsilon_num_frame": epsilon_frame,
        "candidate_budget": budget,
        "one_e_minus_four_sensitivity_only": {
            "paired_observation_count_at_or_below_abs_margin": low_margin_action_count,
            "used_for_selection_acceptance_or_gate": False,
        },
    }
    atomic_write_json(output_dir / "calibration_margins.json", margins)
    manifest = {
        "schema": "visfuse3d_stage019_s2_calibration_aggregate_v1",
        "status": status,
        "protocol_profile": args.protocol_profile,
        "repeat_profile": args.repeat_profile,
        "expected_repeat_ids": list(expected_ids),
        "s2a_reporting_conditions": list(s2a_conditions),
        "actionable_conditions": list(actionable_conditions),
        "hard_bypass_conditions": list(hard_bypass_conditions),
        "condition_destination_applicability": applicability,
        "claim_boundary": calibration_claim_boundary,
        "canonical_repeat_id": args.canonical_repeat_id,
        "repeat_manifests": [
            {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "repeat_id": manifest["repeat_id"],
                "gpu_identity": manifest["gpu_identity"],
            }
            for _, path, manifest in manifests
        ],
        "frame_count_by_condition": frame_count_by_condition,
        "paired_frame_condition_count": int(sum(frame_count_by_condition.values())),
        "proxy_audit_count": {
            condition: proxy_audit_count[condition].tolist()
            for condition in actionable_conditions
        },
        "frame_action_audit_count": frame_action_audit_count,
        "missing_condition_destination_audits": missing_proxy_cells,
        "positive_robust_action_count": positive_action_count,
        "locked_source_identity": locked_identity,
        "source_calibration_root": (
            str(source_calibration_root) if source_calibration_root else None
        ),
        "source_repeat_manifest_sha256": {
            repeat_id: expected_hashes[repeat_id]
            for repeat_id in expected_ids
            if repeat_id in expected_hashes
        },
        "source_reuse_policy": (
            "read_only_raw_trace_and_npz_only_no_prior_margins_or_diagnostic_K"
            if args.protocol_profile == S3_PROTOCOL_PROFILE
            else "native_profile"
        ),
        "margins_path": str((output_dir / "calibration_margins.json").resolve()),
        "margins_sha256": sha256_file(output_dir / "calibration_margins.json"),
    }
    atomic_write_json(output_dir / "calibration_aggregate_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0 if status in {"passed", "s2b_stopped_zero_positive_quality", "candidate_mass_not_coverable_under_K16"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
