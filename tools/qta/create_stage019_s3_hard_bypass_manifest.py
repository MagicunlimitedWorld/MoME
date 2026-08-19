"""Lock an exact Stage019-S3 complete-zero prediction identity bypass."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .common import atomic_write_json
    from .extract_route_loss_cache import is_relative_to, sha256_file
    from .stage019_s2_oracle_worker import S2A_ROLES, S3_PROTOCOL_PROFILE
    from .stage019_s2_training_conditions import S3_HARD_BYPASS_CONDITIONS
except ImportError:
    from common import atomic_write_json
    from extract_route_loss_cache import is_relative_to, sha256_file
    from stage019_s2_oracle_worker import S2A_ROLES, S3_PROTOCOL_PROFILE
    from stage019_s2_training_conditions import S3_HARD_BYPASS_CONDITIONS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--condition", choices=S3_HARD_BYPASS_CONDITIONS, required=True
    )
    parser.add_argument("--s3a-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    artifact_root = args.artifact_root.resolve()
    source_root = Path(__file__).resolve().parents[2]
    if not is_relative_to(output_dir, artifact_root):
        raise ValueError("hard-bypass output must stay inside the S3 artifact root")
    if is_relative_to(output_dir, source_root):
        raise ValueError("hard-bypass output cannot enter the source checkout")
    if output_dir.exists():
        raise FileExistsError(f"immutable hard-bypass output exists: {output_dir}")

    source_path = args.s3a_manifest.resolve()
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if (
        source.get("status") != "complete_official_evaluation"
        or source.get("mode") != "s2a"
        or source.get("condition") != args.condition
        or source.get("protocol_profile") != S3_PROTOCOL_PROFILE
        or int(source.get("scene_count", -1)) != 150
        or int(source.get("frame_count", -1)) != 6019
        or int(source.get("unique_token_count", -1)) != 6019
        or int(source.get("engineering_failure_count", -1)) != 0
    ):
        raise ValueError("S3-A source manifest is not a complete matching zero condition")

    original_role = S2A_ROLES[0]
    prediction = source.get("prediction_outputs", {}).get(original_role)
    if not isinstance(prediction, dict):
        raise ValueError("S3-A source lacks the canonical original prediction output")
    prediction_path = Path(str(prediction.get("path", ""))).resolve()
    prediction_sha256 = str(prediction.get("sha256", "")).upper()
    if not prediction_path.is_file() or sha256_file(prediction_path).upper() != prediction_sha256:
        raise ValueError("S3-A original prediction output hash drifted")
    original_metrics = source.get("metrics", {}).get(original_role)
    if not isinstance(original_metrics, dict):
        raise ValueError("S3-A source lacks original MoME metrics")

    identity_output = {
        "path": str(prediction_path),
        "sha256": prediction_sha256,
        "size_bytes": prediction_path.stat().st_size,
    }
    manifest = {
        "schema": "visfuse3d_stage019_s3_hard_bypass_identity_v1",
        "status": "complete_hard_bypass_identity",
        "protocol_profile": S3_PROTOCOL_PROFILE,
        "condition": args.condition,
        "mode": "s3b_hard_bypass",
        "applicability": "hard_bypass_not_applicable",
        "scene_count": 150,
        "frame_count": 6019,
        "unique_token_count": 6019,
        "engineering_failure_count": 0,
        "base_prediction": identity_output,
        "final_prediction": dict(identity_output),
        "prediction_sha256_identical": True,
        "metrics": {
            "same_run_original_mome": original_metrics,
            "hard_bypass_result": original_metrics,
        },
        "delta": {"hard_bypass_minus_same_run_original": {"mAP": 0.0, "NDS": 0.0}},
        "source_s3a_manifest": {
            "path": str(source_path),
            "sha256": sha256_file(source_path),
        },
        "claim_boundary": (
            "identity_bypass_for_complete_zero_condition_no_route_search_or_margin_audit"
        ),
    }
    output_dir.mkdir(parents=True)
    output_path = output_dir / "bypass_identity_manifest.json"
    atomic_write_json(output_path, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
