#!/usr/bin/env python3
"""Convert raw JAKA LeRobot v2.1 data to a canonical LabVLA dataset.

Canonical arm prefix (shared by arm-only and mobile variants):
    [joint_1..joint_6, reserved_zero, gripper_openness]

Optional base state/action channels are appended after this prefix. They are
never inferred from ``observation.agv`` or from action width: callers must
provide both source keys and explicit indices for a mobile dataset.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# Keep direct ``python scripts/prepare_jaka_dataset.py`` execution independent
# of the caller's current working directory.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_process.stats.running import RunningStats


ARM_DIM = 6
CANONICAL_GRIPPER_DIM = 7


def _parse_indices(raw: str | None, *, name: str) -> list[int] | None:
    if raw is None or not raw.strip():
        return None
    try:
        indices = [int(x.strip()) for x in raw.split(",") if x.strip()]
    except ValueError as exc:
        raise ValueError(f"{name} must be comma-separated integers, got {raw!r}") from exc
    if not indices or any(i < 0 for i in indices):
        raise ValueError(f"{name} must contain at least one non-negative index")
    if len(set(indices)) != len(indices):
        raise ValueError(f"{name} contains duplicate indices: {indices}")
    return indices


def _as_matrix(values: Iterable, *, key: str, rows: int | None = None) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(f"{key!r} must be a per-frame matrix, got shape {matrix.shape}")
    if rows is not None and matrix.shape[0] != rows:
        raise ValueError(f"{key!r} has {matrix.shape[0]} rows, expected {rows}")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{key!r} contains NaN or Inf")
    return matrix


def _select(matrix: np.ndarray, indices: list[int], *, key: str) -> np.ndarray:
    if max(indices) >= matrix.shape[1]:
        raise ValueError(
            f"{key!r} has width {matrix.shape[1]}, but requested indices {indices}"
        )
    return matrix[:, indices]


def _feature_names(info: dict, key: str, width: int) -> list[str]:
    names = ((info.get("features") or {}).get(key) or {}).get("names") or []
    if len(names) == width:
        return [str(x) for x in names]
    return [f"{key.replace('.', '_')}_{i}" for i in range(width)]


def _copy_tree(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(f"missing dataset directory: {src}")

    def copy_or_link(src_file: str, dst_file: str) -> str:
        try:
            os.link(src_file, dst_file)
        except OSError as exc:
            if exc.errno != getattr(os, "EXDEV", 18):
                raise
            shutil.copy2(src_file, dst_file)
        return dst_file

    shutil.copytree(src, dst, copy_function=copy_or_link)


def _build_manifest(*, include_base: bool, base_dim: int) -> dict:
    total_dim = 8 + base_dim
    delta_mask = [True] * ARM_DIM + [False, False]
    if include_base:
        delta_mask.extend([True] * base_dim)
    return {
        "version": 1,
        "schema_id": "jaka_v21_canonical_with_base" if include_base else "jaka_v21_canonical",
        "robot_type": "jaka+base" if include_base else "jaka",
        "state": {"keys": ["observation.state"], "dims": [total_dim]},
        "action": {
            "keys": ["action"],
            "dims": [total_dim],
            "delta": delta_mask,
            "gripper_dims": [CANONICAL_GRIPPER_DIM],
        },
        "images": {"observation.images.front": "image0"},
        "arm_layout": {
            "arm_count": "single",
            "arm_dof": 7,
            "gripper_index_in_raw": CANONICAL_GRIPPER_DIM,
        },
        "gripper_semantic": "open_fraction",
    }


def _stats_bundle(state_stats: RunningStats, action_stats: RunningStats, abs_stats: RunningStats) -> dict:
    def encode(stats: RunningStats) -> dict:
        value = stats.get_statistics()
        out = value.to_json()
        return out

    return {
        "observation.state": encode(state_stats),
        "action": encode(action_stats),
        "action_abs": encode(abs_stats),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, required=True, help="Raw LeRobot v2.1 dataset")
    ap.add_argument("--output", type=Path, required=True, help="Canonical dataset to create")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--chunk_size", type=int, default=50)
    ap.add_argument("--gripper_state_index", type=int, default=1,
                    help="Index of gripper_openness in observation.gripper (default: 1)")
    ap.add_argument("--base_state_key", default=None,
                    help="Optional base state column; requires --base_state_indices")
    ap.add_argument("--base_state_indices", default=None,
                    help="Indices from --base_state_key, e.g. 0,1,2")
    ap.add_argument("--base_action_key", default=None,
                    help="Optional base action column; requires --base_action_indices")
    ap.add_argument("--base_action_indices", default=None,
                    help="Indices from --base_action_key, e.g. 0,1,2")
    ap.add_argument("--base_action_mode", choices=["delta", "abs"], default=None,
                    help="Required when base channels are enabled")
    args = ap.parse_args()

    if args.chunk_size <= 0:
        raise ValueError("--chunk_size must be positive")
    if args.gripper_state_index < 0:
        raise ValueError("--gripper_state_index must be non-negative")

    base_state_indices = _parse_indices(args.base_state_indices, name="--base_state_indices")
    base_action_indices = _parse_indices(args.base_action_indices, name="--base_action_indices")
    base_fields = (args.base_state_key, base_state_indices, args.base_action_key, base_action_indices)
    if any(x is not None for x in base_fields) and not all(x is not None for x in base_fields):
        raise ValueError(
            "base processing requires --base_state_key, --base_state_indices, "
            "--base_action_key, and --base_action_indices together"
        )
    include_base = all(x is not None for x in base_fields)
    if include_base and args.base_action_mode is None:
        raise ValueError("--base_action_mode is required when base channels are enabled")
    if not include_base and args.base_action_mode is not None:
        raise ValueError("--base_action_mode requires base state/action mappings")

    source = args.source.resolve()
    output = args.output.resolve()
    if not (source / "meta" / "info.json").is_file():
        raise FileNotFoundError(f"source meta/info.json not found: {source}")
    if output.exists():
        if not args.force:
            raise FileExistsError(f"output exists: {output}; pass --force to replace it")
        shutil.rmtree(output)
    output.mkdir(parents=True)

    info = json.loads((source / "meta" / "info.json").read_text())
    features = info.get("features") or {}
    required = {"observation.joints", "observation.gripper", "action"}
    missing = sorted(required - set(features))
    if missing:
        raise ValueError(f"source info.json missing required features: {missing}")
    if args.gripper_state_index >= int(features["observation.gripper"]["shape"][0]):
        raise ValueError("--gripper_state_index is outside observation.gripper")

    if include_base:
        for key in (args.base_state_key, args.base_action_key):
            if key not in features:
                raise ValueError(f"base source key {key!r} is not declared in info.json")

    data_out = output / "data"
    data_out.mkdir()
    videos_src = source / "videos"
    if videos_src.exists():
        _copy_tree(videos_src, output / "videos")
    meta_out = output / "meta"
    shutil.copytree(source / "meta", meta_out, dirs_exist_ok=True)

    base_dim = len(base_state_indices or [])
    if include_base and base_dim != len(base_action_indices or []):
        raise ValueError("base state and base action dimensions must match")
    total_dim = 8 + base_dim
    state_names = [f"joint_{i + 1}" for i in range(ARM_DIM)] + [
        "reserved_arm_slot", "gripper_openness"
    ]
    action_names = list(state_names)
    if include_base:
        base_names = [f"base_{i}" for i in range(base_dim)]
        state_names.extend(base_names)
        action_names.extend(base_names)

    state_running = RunningStats()
    action_running = RunningStats()
    abs_running = RunningStats()
    total_rows = 0
    parquet_files = sorted((source / "data").glob("chunk-*/episode_*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"no episode parquet files under {source / 'data'}")

    for src_file in parquet_files:
        table = pq.read_table(src_file)
        rows = table.num_rows
        joints = _as_matrix(table["observation.joints"].to_pylist(), key="observation.joints", rows=rows)
        gripper = _as_matrix(table["observation.gripper"].to_pylist(), key="observation.gripper", rows=rows)
        raw_action = _as_matrix(table["action"].to_pylist(), key="action", rows=rows)
        if joints.shape[1] != ARM_DIM:
            raise ValueError(f"{src_file}: observation.joints width must be 6, got {joints.shape[1]}")
        if raw_action.shape[1] < ARM_DIM + 1:
            raise ValueError(f"{src_file}: action width must be at least 7, got {raw_action.shape[1]}")

        state = np.concatenate(
            [joints, np.zeros((rows, 1), dtype=np.float32), gripper[:, args.gripper_state_index:args.gripper_state_index + 1]],
            axis=1,
        )
        action = np.concatenate(
            [raw_action[:, :ARM_DIM], np.zeros((rows, 1), dtype=np.float32), raw_action[:, ARM_DIM:ARM_DIM + 1]],
            axis=1,
        )
        if include_base:
            base_state = _select(_as_matrix(table[args.base_state_key].to_pylist(), key=args.base_state_key, rows=rows), base_state_indices, key=args.base_state_key)
            base_action = _select(_as_matrix(table[args.base_action_key].to_pylist(), key=args.base_action_key, rows=rows), base_action_indices, key=args.base_action_key)
            state = np.concatenate([state, base_state], axis=1)
            action = np.concatenate([action, base_action], axis=1)

        state_running.update(state)
        abs_running.update(action)
        delta_mask = np.array([True] * ARM_DIM + [False, False] + ([args.base_action_mode == "delta"] * base_dim if include_base else []), dtype=bool)
        for start in range(rows):
            end = min(rows, start + args.chunk_size)
            chunk = action[start:end].copy()
            chunk[:, delta_mask] -= state[start, delta_mask]
            action_running.update(chunk)

        columns = {name: table[name] for name in table.column_names}
        columns["observation.state"] = pa.array(state.tolist(), type=pa.list_(pa.float32()))
        columns["action"] = pa.array(action.tolist(), type=pa.list_(pa.float32()))
        rel = src_file.relative_to(source / "data")
        dst_file = data_out / rel
        dst_file.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.table(columns), dst_file, compression="snappy")
        total_rows += rows

    expected = int(info.get("total_frames", total_rows))
    if total_rows != expected:
        raise ValueError(f"rewrote {total_rows} frames but info.json declares {expected}")

    output_info = dict(info)
    output_info["robot_type"] = "jaka+base" if include_base else "jaka"
    output_info["features"] = dict(features)
    output_info["features"]["observation.state"] = {
        "dtype": "float32", "shape": [total_dim], "names": state_names,
    }
    output_info["features"]["action"] = {
        "dtype": "float32", "shape": [total_dim], "names": action_names,
    }
    (meta_out / "info.json").write_text(json.dumps(output_info, indent=2) + "\n")
    manifest = _build_manifest(include_base=include_base, base_dim=base_dim)
    (meta_out / "labvla_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    stats = _stats_bundle(state_running, action_running, abs_running)
    stats["_chunk_size"] = args.chunk_size
    (meta_out / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")

    mode = "with_base" if include_base else "arm_only"
    print(f"prepared {output} ({mode}): frames={total_rows} state_dim={total_dim} action_dim={total_dim}")
    print(f"manifest: {meta_out / 'labvla_manifest.json'}")
    print(f"stats: {meta_out / 'stats.json'}")


if __name__ == "__main__":
    main()
