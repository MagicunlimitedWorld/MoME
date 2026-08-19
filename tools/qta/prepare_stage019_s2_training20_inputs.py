"""Derive immutable training20 and 4-beam inputs inside the S2 artifact root."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path

try:
    from .common import atomic_write_json
    from .extract_route_loss_cache import is_relative_to, sha256_file
except ImportError:
    from common import atomic_write_json
    from extract_route_loss_cache import is_relative_to, sha256_file


EXPECTED_TRAIN_ANNOTATION_SHA256 = (
    "9214A60ADF903647DF027FE2F49CFBBB3CDF14DC4CE2C0B099337F7E92AB34A6"
)
EXPECTED_SCENE_SPLIT_SHA256 = (
    "0C1C9581B128FFFEB03E456846FA12548498F5A1494EFC03FCFE46C85567DF97"
)
EXPECTED_SAMPLE_MAP_SHA256 = (
    "D9D129B3600AC57F30A7B978D6E251AFA84C01A68646EC8BF1B9BE011680EB24"
)
EXPECTED_SELECTION_MANIFEST_SHA256 = (
    "22794274BCB0121417F563534870BE640861F1359F85519E89706066DEAE8B9F"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-annotation", type=Path, required=True)
    parser.add_argument("--scene-split", type=Path, required=True)
    parser.add_argument("--sample-scene-map", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--adapter-scripts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    return parser.parse_args()


def _tree_sha256(root: Path) -> tuple[str, list[dict]]:
    records = []
    lines = []
    for path in sorted(
        (item for item in root.rglob("*") if item.is_file()),
        key=lambda item: item.relative_to(root).as_posix(),
    ):
        relative = path.relative_to(root).as_posix()
        digest = sha256_file(path)
        size = path.stat().st_size
        lines.append(f"{relative}|{size}|{digest}")
        records.append(
            {"relative_path": relative, "size": size, "sha256": digest}
        )
    payload = "\n".join(lines).encode("utf-8")
    return hashlib.sha256(payload).hexdigest().upper(), records


def _token_set_sha256(tokens: list[str]) -> str:
    payload = "".join(f"{token}\n" for token in sorted(set(tokens)))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest().upper()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    artifact_root = args.artifact_root.resolve()
    if not is_relative_to(output_dir, artifact_root):
        raise ValueError("training20 inputs must stay inside the S2 artifact root")
    if output_dir.exists():
        raise FileExistsError(f"immutable output already exists: {output_dir}")
    expected_hashes = {
        args.train_annotation: EXPECTED_TRAIN_ANNOTATION_SHA256,
        args.scene_split: EXPECTED_SCENE_SPLIT_SHA256,
        args.sample_scene_map: EXPECTED_SAMPLE_MAP_SHA256,
        args.selection_manifest: EXPECTED_SELECTION_MANIFEST_SHA256,
    }
    for path, expected in expected_hashes.items():
        if sha256_file(path) != expected:
            raise ValueError(f"locked source SHA256 mismatch: {path}")

    adapter_scripts = args.adapter_scripts.resolve()
    if str(adapter_scripts) not in sys.path:
        sys.path.insert(0, str(adapter_scripts))
    from nuscenes_r_mome_conditions import (
        _dump_cross_environment_pickle,
        build_beam4_overlay,
        collect_lidar_paths,
        derive_beam4_info,
    )

    selection = json.loads(args.selection_manifest.read_text(encoding="utf-8"))
    scenes = selection.get("input_identity", {}).get("requested_scenes")
    if not isinstance(scenes, list) or len(scenes) != 20 or len(set(scenes)) != 20:
        raise ValueError("training20 selection manifest must lock 20 unique scenes")
    sample_map = json.loads(args.sample_scene_map.read_text(encoding="utf-8"))
    sample_to_scene = {
        str(token): str(scene)
        for token, scene in sample_map["sample_to_scene"].items()
    }
    selected_scenes = set(str(scene) for scene in scenes)

    annotation_hash_before = sha256_file(args.train_annotation)
    with args.train_annotation.open("rb") as stream:
        source_payload = pickle.load(stream)  # noqa: S301 - locked local input.
    infos = source_payload.get("infos") if isinstance(source_payload, dict) else None
    if not isinstance(infos, list):
        raise ValueError("train annotation lacks infos list")
    selected_infos = [
        info
        for info in infos
        if sample_to_scene.get(str(info["token"])) in selected_scenes
    ]
    tokens = [str(info["token"]) for info in selected_infos]
    if len(tokens) != 803 or len(set(tokens)) != 803:
        raise RuntimeError(
            f"training20 annotation must contain 803 unique frames, got {len(tokens)}"
        )
    observed_scenes = {sample_to_scene[token] for token in tokens}
    if observed_scenes != selected_scenes:
        raise RuntimeError("training20 annotation scene set drifted")

    output_dir.mkdir(parents=True)
    clean_annotation = output_dir / "training20_clean.pkl"
    subset_payload = dict(source_payload)
    subset_payload["infos"] = selected_infos
    with clean_annotation.open("wb") as stream:
        _dump_cross_environment_pickle(subset_payload, stream)

    source_paths = collect_lidar_paths(subset_payload, args.dataset_root)
    overlay_root = output_dir / "beam4_overlay"
    mapping, overlay_records = build_beam4_overlay(
        source_paths,
        dataset_root=args.dataset_root,
        overlay_root=overlay_root,
    )
    beam_annotation = output_dir / "training20_beam4.pkl"
    beam_annotation_record = derive_beam4_info(
        clean_annotation,
        beam_annotation,
        mapping=mapping,
        dataset_root=args.dataset_root,
    )
    tree_hash, tree_records = _tree_sha256(overlay_root)
    if len(tree_records) != len(source_paths):
        raise RuntimeError("beam overlay tree file count drifted")
    annotation_hash_after = sha256_file(args.train_annotation)
    if annotation_hash_after != annotation_hash_before:
        raise RuntimeError("source train annotation changed during derivation")

    per_file_manifest = {
        "schema": "visfuse3d_stage019_s2_training20_beam4_file_manifest_v1",
        "file_count": len(tree_records),
        "tree_hash_algorithm": (
            "sorted_relative_path_forward_slash_pipe_size_pipe_file_sha256_"
            "joined_with_LF_no_trailing_LF"
        ),
        "tree_sha256": tree_hash,
        "files": tree_records,
    }
    atomic_write_json(output_dir / "beam4_file_manifest.json", per_file_manifest)
    manifest = {
        "schema": "visfuse3d_stage019_s2_training20_inputs_manifest_v1",
        "status": "passed",
        "scene_count": 20,
        "frame_count": 803,
        "scene_tokens": scenes,
        "frame_token_set_sha256": _token_set_sha256(tokens),
        "source_train_annotation": str(args.train_annotation.resolve()),
        "source_train_annotation_sha256_before": annotation_hash_before,
        "source_train_annotation_sha256_after": annotation_hash_after,
        "source_unchanged": annotation_hash_before == annotation_hash_after,
        "clean_annotation": str(clean_annotation.resolve()),
        "clean_annotation_sha256": sha256_file(clean_annotation),
        "beam4_annotation": str(beam_annotation.resolve()),
        "beam4_annotation_sha256": sha256_file(beam_annotation),
        "beam4_source_info": beam_annotation_record,
        "beam4_overlay_root": str(overlay_root.resolve()),
        "beam4_unique_source_file_count": len(source_paths),
        "beam4_overlay_tree_sha256": tree_hash,
        "beam4_overlay_records_count": len(overlay_records),
        "scene_split_sha256": sha256_file(args.scene_split),
        "sample_scene_map_sha256": sha256_file(args.sample_scene_map),
        "selection_manifest_sha256": sha256_file(args.selection_manifest),
        "persistent_outputs_within_artifact_root": True,
    }
    atomic_write_json(output_dir / "training20_inputs_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
