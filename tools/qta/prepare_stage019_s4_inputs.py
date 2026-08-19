"""Prepare immutable train-only Stage019-S4 fit/calibration inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path
from typing import Any

try:
    from .common import atomic_write_json
    from .extract_route_loss_cache import is_relative_to
    from .stage019_s4_contracts import (
        PROTOCOL_PROFILE,
        RUN_ID,
        SAMPLE_SCENE_MAP_SHA256,
        SCENE_SPLIT_SHA256,
        STAGE_ID,
        TRAIN_ANNOTATION_SHA256,
        TRAINING20_INPUTS_MANIFEST_SHA256,
        scene_serialization_sha256,
        sha256_file,
        validate_train_scene_split,
    )
except ImportError:
    from common import atomic_write_json
    from extract_route_loss_cache import is_relative_to
    from stage019_s4_contracts import (
        PROTOCOL_PROFILE,
        RUN_ID,
        SAMPLE_SCENE_MAP_SHA256,
        SCENE_SPLIT_SHA256,
        STAGE_ID,
        TRAIN_ANNOTATION_SHA256,
        TRAINING20_INPUTS_MANIFEST_SHA256,
        scene_serialization_sha256,
        sha256_file,
        validate_train_scene_split,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-annotation", type=Path, required=True)
    parser.add_argument("--scene-split", type=Path, required=True)
    parser.add_argument("--sample-scene-map", type=Path, required=True)
    parser.add_argument("--training20-inputs-manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--adapter-scripts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    return parser.parse_args()


def _token_set_sha256(tokens: list[str]) -> str:
    payload = "".join(f"{token}\n" for token in sorted(set(tokens)))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest().upper()


def _tree_manifest(root: Path) -> tuple[str, list[dict[str, Any]]]:
    records = []
    lines = []
    for path in sorted(
        (value for value in root.rglob("*") if value.is_file()),
        key=lambda value: value.relative_to(root).as_posix(),
    ):
        relative = path.relative_to(root).as_posix()
        size = path.stat().st_size
        digest = sha256_file(path)
        records.append({"relative_path": relative, "size": size, "sha256": digest})
        lines.append(f"{relative}|{size}|{digest}")
    tree = hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest().upper()
    return tree, records


def _partition_record(
    name: str,
    scenes: list[str],
    infos: list[dict[str, Any]],
    clean_annotation: Path,
    beam_annotation: Path,
    overlay_root: Path,
    overlay_tree_sha256: str,
    overlay_file_count: int,
) -> dict[str, Any]:
    tokens = [str(info["token"]) for info in infos]
    if len(tokens) != len(set(tokens)):
        raise ValueError(f"{name} annotation contains duplicate frame tokens")
    return {
        "scene_tokens": scenes,
        "scene_count": len(scenes),
        "scene_serialization_sha256": scene_serialization_sha256(scenes),
        "frame_count": len(tokens),
        "frame_token_set_sha256": _token_set_sha256(tokens),
        "clean_annotation": {
            "path": str(clean_annotation.resolve()),
            "sha256": sha256_file(clean_annotation),
        },
        "beam4_annotation": {
            "path": str(beam_annotation.resolve()),
            "sha256": sha256_file(beam_annotation),
        },
        "beam4_overlay": {
            "root": str(overlay_root.resolve()),
            "tree_sha256": overlay_tree_sha256,
            "file_count": overlay_file_count,
        },
    }


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    artifact_root = args.artifact_root.resolve()
    source_root = Path(__file__).resolve().parents[2]
    if not is_relative_to(output_dir, artifact_root):
        raise ValueError("S4 inputs must stay inside the artifact root")
    if is_relative_to(output_dir, source_root):
        raise ValueError("S4 inputs cannot enter the source checkout")
    if output_dir.exists():
        raise FileExistsError(f"immutable S4 input directory exists: {output_dir}")
    locked = {
        args.train_annotation: TRAIN_ANNOTATION_SHA256,
        args.scene_split: SCENE_SPLIT_SHA256,
        args.sample_scene_map: SAMPLE_SCENE_MAP_SHA256,
        args.training20_inputs_manifest: TRAINING20_INPUTS_MANIFEST_SHA256,
    }
    for path, expected in locked.items():
        if sha256_file(path) != expected:
            raise ValueError(f"locked S4 source SHA256 mismatch: {path}")
    training20_manifest = json.loads(
        args.training20_inputs_manifest.read_text(encoding="utf-8")
    )
    if (
        training20_manifest.get("status") != "passed"
        or int(training20_manifest.get("scene_count", -1)) != 20
        or int(training20_manifest.get("frame_count", -1)) != 803
    ):
        raise ValueError("locked training20 input manifest is incomplete")
    for key, hash_key in (
        ("clean_annotation", "clean_annotation_sha256"),
        ("beam4_annotation", "beam4_annotation_sha256"),
    ):
        path = Path(str(training20_manifest[key]))
        if not path.is_file() or sha256_file(path) != training20_manifest[hash_key]:
            raise ValueError("training20 referenced input is missing or changed")

    split = json.loads(args.scene_split.read_text(encoding="utf-8"))
    fit80 = [str(value) for value in split["stage019a_training_scenes"]]
    calibration20 = [str(value) for value in split["stage019a_test_scenes"]]
    training20 = [str(value) for value in training20_manifest["scene_tokens"]]
    isolation = validate_train_scene_split(fit80, calibration20, training20)
    fit60 = [scene for scene in fit80 if scene not in set(training20)]
    if len(fit60) != 60:
        raise ValueError("fit80 minus locked training20 must contain 60 scenes")

    sample_map_payload = json.loads(args.sample_scene_map.read_text(encoding="utf-8"))
    sample_to_scene = {
        str(token): str(scene)
        for token, scene in sample_map_payload["sample_to_scene"].items()
    }
    source_before = sha256_file(args.train_annotation)
    with args.train_annotation.open("rb") as stream:
        payload = pickle.load(stream)  # noqa: S301 - locked local source.
    infos = payload.get("infos") if isinstance(payload, dict) else None
    if not isinstance(infos, list):
        raise ValueError("locked train annotation lacks an infos list")

    scene_groups = {
        "fit80": fit80,
        "fit60_remaining": fit60,
        "calibration20": calibration20,
    }
    partition_infos = {
        name: [
            info
            for info in infos
            if sample_to_scene.get(str(info["token"])) in set(scenes)
        ]
        for name, scenes in scene_groups.items()
    }
    for name, scenes in scene_groups.items():
        observed = {
            sample_to_scene[str(info["token"])] for info in partition_infos[name]
        }
        if observed != set(scenes):
            raise ValueError(f"{name} annotation does not cover its exact scene set")

    adapter_scripts = args.adapter_scripts.resolve()
    if str(adapter_scripts) not in sys.path:
        sys.path.insert(0, str(adapter_scripts))
    from nuscenes_r_mome_conditions import (
        _dump_cross_environment_pickle,
        build_beam4_overlay,
        collect_lidar_paths,
        derive_beam4_info,
    )

    output_dir.mkdir(parents=True)
    clean_paths: dict[str, Path] = {}
    for name, selected in partition_infos.items():
        clean_path = output_dir / f"{name}_clean.pkl"
        selected_payload = dict(payload)
        selected_payload["infos"] = selected
        with clean_path.open("wb") as stream:
            _dump_cross_environment_pickle(selected_payload, stream)
        clean_paths[name] = clean_path

    union_payload = dict(payload)
    union_scene_set = set(fit80) | set(calibration20)
    union_payload["infos"] = [
        info
        for info in infos
        if sample_to_scene.get(str(info["token"])) in union_scene_set
    ]
    source_paths = collect_lidar_paths(union_payload, args.dataset_root)
    overlay_root = output_dir / "beam4_overlay_fit80_calibration20"
    mapping, overlay_records = build_beam4_overlay(
        source_paths, dataset_root=args.dataset_root, overlay_root=overlay_root
    )
    beam_paths: dict[str, Path] = {}
    beam_source_records = {}
    for name, clean_path in clean_paths.items():
        beam_path = output_dir / f"{name}_beam4.pkl"
        beam_source_records[name] = derive_beam4_info(
            clean_path,
            beam_path,
            mapping=mapping,
            dataset_root=args.dataset_root,
        )
        beam_paths[name] = beam_path
    tree_hash, tree_records = _tree_manifest(overlay_root)
    if len(tree_records) != len(source_paths) or len(overlay_records) != len(source_paths):
        raise RuntimeError("S4 beam4 overlay file coverage drifted")
    atomic_write_json(
        output_dir / "stage019_s4_beam4_file_manifest.json",
        {
            "schema": "visfuse3d_stage019_s4_beam4_file_manifest_v1",
            "tree_hash_algorithm": (
                "sorted_relative_path_forward_slash_pipe_size_pipe_file_sha256_"
                "joined_with_LF_no_trailing_LF"
            ),
            "tree_sha256": tree_hash,
            "file_count": len(tree_records),
            "files": tree_records,
        },
    )
    partitions = {
        name: _partition_record(
            name,
            scenes,
            partition_infos[name],
            clean_paths[name],
            beam_paths[name],
            overlay_root,
            tree_hash,
            len(tree_records),
        )
        for name, scenes in scene_groups.items()
    }
    partitions["training20_reuse"] = {
        "scene_tokens": training20,
        "scene_count": 20,
        "scene_serialization_sha256": scene_serialization_sha256(training20),
        "frame_count": 803,
        "frame_token_set_sha256": training20_manifest["frame_token_set_sha256"],
        "source_manifest": {
            "path": str(args.training20_inputs_manifest.resolve()),
            "sha256": TRAINING20_INPUTS_MANIFEST_SHA256,
        },
        "clean_annotation": {
            "path": str(Path(training20_manifest["clean_annotation"]).resolve()),
            "sha256": training20_manifest["clean_annotation_sha256"],
        },
        "beam4_annotation": {
            "path": str(Path(training20_manifest["beam4_annotation"]).resolve()),
            "sha256": training20_manifest["beam4_annotation_sha256"],
        },
        "beam4_overlay": {
            "root": str(Path(training20_manifest["beam4_overlay_root"]).resolve()),
            "tree_sha256": training20_manifest["beam4_overlay_tree_sha256"],
            "file_count": int(training20_manifest["beam4_unique_source_file_count"]),
        },
    }
    source_after = sha256_file(args.train_annotation)
    if source_after != source_before:
        raise RuntimeError("source train annotation changed during S4 preparation")
    manifest = {
        "schema": "visfuse3d_stage019_s4_inputs_manifest_v1",
        "status": "passed",
        "stage_id": STAGE_ID,
        "protocol_profile": PROTOCOL_PROFILE,
        "run_id": RUN_ID,
        "locked_sources": {
            "train_annotation": {
                "path": str(args.train_annotation.resolve()),
                "sha256": source_before,
            },
            "scene_split": {
                "path": str(args.scene_split.resolve()),
                "sha256": SCENE_SPLIT_SHA256,
            },
            "sample_scene_map": {
                "path": str(args.sample_scene_map.resolve()),
                "sha256": SAMPLE_SCENE_MAP_SHA256,
            },
            "training20_inputs_manifest": {
                "path": str(args.training20_inputs_manifest.resolve()),
                "sha256": TRAINING20_INPUTS_MANIFEST_SHA256,
            },
        },
        "partitions": partitions,
        "isolation": isolation,
        "beam4_source_records": beam_source_records,
        "source_unchanged": source_before == source_after,
        "persistent_outputs_within_artifact_root": True,
    }
    atomic_write_json(output_dir / "stage019_s4_inputs_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
