#!/usr/bin/env python3
"""Convert raw JAKA RGB3 episodes to LabVLA's LeRobot v2.1 layout.

The raw recording may contain front, side and wrist videos. The first camera
selected by ``--cameras`` defines the output timeline; other selected cameras
are aligned by nearest timestamp. State is linearly interpolated at the
selected primary timestamps and the action is the next selected state.

Without ``--mobile`` this writes arm-only 8-D datasets. With ``--mobile`` it
writes JAKA + AGV 10-D datasets and preserves ``observation.agv``.

By default this writes both rates:
  <parent>/<prefix>_30hz
  <parent>/<prefix>_10hz

``actions.csv`` is intentionally not read.  The current raw recorder writes
only its header there, while the existing JAKA conversion path uses the next
aligned state as the action target.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


CAMERA_SPECS = {
    "front": ("frames.csv", "video.mp4"),
    "side": ("frames_side.csv", "video_side.mp4"),
    "wrist": ("frames_wrist.csv", "video_wrist.mp4"),
}
JOINT_KEYS = [f"joint_{i}_deg" for i in range(1, 7)]
TCP_KEYS = [
    "tcp_x_mm", "tcp_y_mm", "tcp_z_mm",
    "tcp_rx_deg", "tcp_ry_deg", "tcp_rz_deg",
]
STATE_KEYS = [*JOINT_KEYS, *TCP_KEYS, "gripper_openness"]
AGV_KEYS = [
    "agv_x", "agv_y", "agv_theta", "agv_linear_m_s",
    "agv_angular_rad_s", "agv_power_percent", "agv_is_moving",
    "agv_charge_state", "agv_estop_state",
]


def parse_camera_list(value: str) -> list[str]:
    cameras = [item.strip().lower() for item in value.split(",") if item.strip()]
    if not 1 <= len(cameras) <= 3:
        raise argparse.ArgumentTypeError("--cameras must contain 1, 2, or 3 cameras")
    if len(set(cameras)) != len(cameras):
        raise argparse.ArgumentTypeError("--cameras must not contain duplicates")
    unknown = sorted(set(cameras) - set(CAMERA_SPECS))
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown camera(s) {unknown}; choose from front,side,wrist"
        )
    return cameras


def camera_config(cameras: list[str]) -> list[tuple[str, str, str, str]]:
    return [
        (name, *CAMERA_SPECS[name], f"observation.images.image{i}")
        for i, name in enumerate(cameras)
    ]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def finite_float(row: dict[str, str], key: str, default: float | None = None) -> float:
    raw = row.get(key, "")
    try:
        value = float(raw)
    except (TypeError, ValueError):
        value = float("nan")
    if np.isfinite(value):
        return value
    if default is not None:
        return float(default)
    raise ValueError(f"non-finite {key} value: {raw!r}")


def state_matrix(
    rows: list[dict[str, str]],
    *,
    missing_gripper: str,
    mobile: bool = False,
) -> tuple[np.ndarray, np.ndarray, int]:
    valid = [r for r in rows if int(r.get("read_ret", "0")) == 0]
    if len(valid) < 2:
        raise RuntimeError("episode needs at least two valid state rows")

    timestamps = np.asarray([int(r["state_timestamp_ns"]) for r in valid], dtype=np.float64)
    order = np.argsort(timestamps)
    timestamps = timestamps[order]
    if np.any(np.diff(timestamps) <= 0):
        raise RuntimeError("state_timestamp_ns must be strictly increasing")

    values = []
    gripper_values = []
    missing_count = 0
    for i in order:
        row = valid[int(i)]
        vals = [finite_float(row, key) for key in [*JOINT_KEYS, *TCP_KEYS]]
        try:
            gripper = finite_float(row, "gripper_openness")
        except ValueError:
            missing_count += 1
            if missing_gripper == "error":
                raise RuntimeError(
                    "gripper_openness contains missing/non-finite values; "
                    "pass --missing-gripper interpolate/zero or collect valid gripper data"
                )
            gripper = float("nan")
        gripper_values.append(gripper)
        agv = [finite_float(row, key, default=0.0) for key in AGV_KEYS] if mobile else []
        values.append([*vals, gripper, *agv])
    if missing_count and missing_gripper == "interpolate":
        gripper_array = np.asarray(gripper_values, dtype=np.float64)
        known = np.isfinite(gripper_array)
        if known.any():
            gripper_array[~known] = np.interp(
                timestamps[~known], timestamps[known], gripper_array[known]
            )
        else:
            gripper_array[:] = 0.0
        for row, gripper in zip(values, gripper_array):
            row[len(JOINT_KEYS) + len(TCP_KEYS)] = float(gripper)
    elif missing_count:
        for row in values:
            gripper_index = len(JOINT_KEYS) + len(TCP_KEYS)
            if not np.isfinite(row[gripper_index]):
                row[gripper_index] = 0.0
    return timestamps, np.asarray(values, dtype=np.float64), missing_count


def interpolate(
    source_t: np.ndarray,
    source_values: np.ndarray,
    target_t: np.ndarray,
) -> np.ndarray:
    return np.column_stack([
        np.interp(target_t, source_t, source_values[:, j])
        for j in range(source_values.shape[1])
    ]).astype(np.float32)


def nearest_indices(source_t: np.ndarray, target_t: np.ndarray) -> list[int]:
    positions = np.searchsorted(source_t, target_t, side="left")
    positions = np.clip(positions, 0, len(source_t) - 1)
    left = np.maximum(positions - 1, 0)
    use_left = np.abs(target_t - source_t[left]) <= np.abs(target_t - source_t[positions])
    return np.where(use_left, left, positions).astype(int).tolist()


def read_selected_video(video_path: Path, selected_indices: list[int]) -> list[np.ndarray]:
    wanted = set(selected_indices)
    frames: dict[int, np.ndarray] = {}
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    source_i = 0
    try:
        while wanted:
            ok, frame = cap.read()
            if not ok:
                break
            if source_i in wanted:
                frames[source_i] = frame
                wanted.remove(source_i)
            source_i += 1
    finally:
        cap.release()
    if wanted:
        raise RuntimeError(
            f"{video_path}: missing source frame indices {sorted(wanted)[:8]}"
        )
    return [frames[i] for i in selected_indices]


def parquet_schema(cameras: list[str], *, mobile: bool = False) -> pa.Schema:
    fields = [
        pa.field("timestamp", pa.float32()),
        pa.field("target_timestamp_ns", pa.int64()),
        pa.field("raw_camera_timestamp_ns", pa.int64()),
        pa.field("raw_video_frame_index", pa.int64()),
        pa.field("frame_index", pa.int64()),
        pa.field("episode_index", pa.int64()),
        pa.field("index", pa.int64()),
        pa.field("task_index", pa.int64()),
        pa.field("observation.state", pa.list_(pa.float32(), 22 if mobile else 13)),
        pa.field("observation.joints", pa.list_(pa.float32(), 6)),
        pa.field("observation.tcp_pose", pa.list_(pa.float32(), 6)),
        pa.field("observation.gripper", pa.list_(pa.float32(), 2)),
        pa.field("observation.agv", pa.list_(pa.float32(), 9)),
        pa.field("action", pa.list_(pa.float32(), 7)),
    ]
    fields.extend(
        pa.field(key, pa.string())
        for _, _, _, key in camera_config(cameras)
    )
    return pa.schema(fields)


def stats(values: list[list[float]]) -> dict:
    x = np.asarray(values, dtype=np.float64)
    return {
        "min": np.min(x, axis=0).tolist(),
        "max": np.max(x, axis=0).tolist(),
        "mean": np.mean(x, axis=0).tolist(),
        "std": np.std(x, axis=0).tolist(),
        "q01": np.quantile(x, 0.01, axis=0).tolist(),
        "q99": np.quantile(x, 0.99, axis=0).tolist(),
        "count": [int(x.shape[0])],
    }


def write_stats(all_rows: list[dict], meta_dir: Path, *, mobile: bool = False) -> None:
    state = np.asarray([
        [*row["observation.joints"], 0.0, row["observation.gripper"][1],
         *([*row["observation.agv"][3:5]] if mobile else [])]
        for row in all_rows
    ], dtype=np.float64)
    # Training consumes 50-step action chunks. Action normalization statistics
    # must therefore be computed over every expanded chunk element relative to
    # that sample's current state, rather than over one parquet row at a time.
    chunk_size = 50
    chunk_abs = []
    chunk_delta = []
    by_episode: dict[int, list[dict]] = {}
    for row in all_rows:
        by_episode.setdefault(int(row["episode_index"]), []).append(row)
    for episode_rows in by_episode.values():
        ep_state = np.asarray([
            [*row["observation.joints"], 0.0, row["observation.gripper"][1],
             *([*row["observation.agv"][3:5]] if mobile else [])]
            for row in episode_rows
        ], dtype=np.float64)
        ep_action_abs = np.asarray([
            [*row["action"][:6], 0.0, row["action"][6],
             *([*row["observation.agv"][3:5]] if mobile else [])]
            for row in episode_rows
        ], dtype=np.float64)
        for start in range(len(episode_rows)):
            indices = np.minimum(
                np.arange(start, start + chunk_size), len(episode_rows) - 1
            )
            actions = ep_action_abs[indices].copy()
            actions[:, :6] -= ep_state[start, :6]
            chunk_delta.append(actions)
            chunk_abs.append(ep_action_abs[indices])
    action = np.concatenate(chunk_delta, axis=0)
    action_abs = np.concatenate(chunk_abs, axis=0)
    canonical = {
        "observation.state": stats(state.tolist()),
        "action": stats(action.tolist()),
        "action_abs": stats(action_abs.tolist()),
    }
    canonical["_chunk_size"] = chunk_size
    stats_name = "stats_labvla_jaka_mobile_10d.json" if mobile else "stats_labvla_jaka_8d.json"
    canonical_text = json.dumps(canonical, indent=2) + "\n"
    # Keep the dataset's standard stats.json canonical as well. The preflight
    # command resolves this file before any external sidecar; writing raw
    # parquet widths here (22-D mobile state / 7-D raw action) would make an
    # otherwise trainable dataset fail schema validation.
    (meta_dir / "stats.json").write_text(canonical_text)
    (meta_dir / stats_name).write_text(
        canonical_text
    )


def convert_episode(
    raw: Path,
    output: Path,
    episode_index: int,
    stride: int,
    global_index: int,
    missing_gripper: str,
    cameras: list[str],
    mobile: bool = False,
) -> tuple[list[dict], dict, int]:
    manifest = json.loads((raw / "manifest.json").read_text())
    camera_rows = {
        name: read_csv(raw / frame_file)
        for name, frame_file, _, _ in camera_config(cameras)
    }
    source_t, source_values, missing_count = state_matrix(
        read_csv(raw / "states.csv"), missing_gripper=missing_gripper, mobile=mobile
    )

    # Use only the common timestamp intersection. Without this guard, a primary
    # frame outside a shorter camera/state stream gets paired with the
    # nearest boundary sample, which can silently create 100-300 ms false
    # synchronization while still producing a valid-looking dataset.
    primary = cameras[0]
    primary_rows = camera_rows[primary]
    camera_t = {
        name: np.asarray(
            [int(row["camera_timestamp_ns"]) for row in rows], dtype=np.float64
        )
        for name, rows in camera_rows.items()
    }
    common_start = max(float(source_t[0]), *(float(t[0]) for t in camera_t.values()))
    common_end = min(float(source_t[-1]), *(float(t[-1]) for t in camera_t.values()))
    aligned_primary = [
        row for row in primary_rows
        if common_start <= int(row["camera_timestamp_ns"]) <= common_end
    ]
    selected_primary = aligned_primary[::stride]
    if not selected_primary:
        raise RuntimeError(
            f"{raw}: no {primary} frames in common camera/state interval"
        )
    target_t = np.asarray(
        [int(row["camera_timestamp_ns"]) for row in selected_primary], dtype=np.float64
    )
    selected_states = interpolate(source_t, source_values, target_t)

    selected_by_camera: dict[str, list[int]] = {}
    for name, rows in camera_rows.items():
        timestamps = np.asarray(
            [int(row["camera_timestamp_ns"]) for row in rows], dtype=np.float64
        )
        selected_by_camera[name] = nearest_indices(timestamps, target_t)

    video_paths = {}
    for name, _, video_file, key in camera_config(cameras):
        video_rel = f"videos/chunk-000/{key}/episode_{episode_index:06d}.mp4"
        video_path = output / video_rel
        video_path.parent.mkdir(parents=True, exist_ok=True)
        selected_frames = read_selected_video(raw / video_file, selected_by_camera[name])
        if len(selected_frames) != len(selected_primary):
            raise RuntimeError(f"{raw}/{video_file}: selected frame count mismatch")
        height, width = selected_frames[0].shape[:2]
        writer = cv2.VideoWriter(
            str(video_path), cv2.VideoWriter_fourcc(*"mp4v"),
            30.0 / stride, (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"cannot create video: {video_path}")
        try:
            for frame in selected_frames:
                writer.write(frame)
        finally:
            writer.release()
        video_paths[name] = video_rel

    rows = []
    for i, (primary_row, state) in enumerate(zip(selected_primary, selected_states)):
        joints = state[:6].tolist()
        tcp = state[6:12].tolist()
        openness = float(state[12])
        agv = state[13:22].tolist() if mobile else [0.0] * 9
        next_state = selected_states[min(i + 1, len(selected_states) - 1)]
        rows.append({
            "timestamp": float(i / (30.0 / stride)),
            "target_timestamp_ns": int(primary_row["camera_timestamp_ns"]),
            "raw_camera_timestamp_ns": int(primary_row["camera_timestamp_ns"]),
            "raw_video_frame_index": int(primary_row["frame_index"]),
            "frame_index": i, "episode_index": episode_index,
            "index": global_index + i, "task_index": 0,
            "observation.state": [
                *joints, *tcp[3:6], *tcp[:3], openness,
                *([*agv] if mobile else []),
            ],
            "observation.joints": joints,
            "observation.tcp_pose": tcp,
            "observation.gripper": [0.0, openness],
            "observation.agv": agv,
            "action": [*next_state[:6].tolist(), float(next_state[12])],
            **{
                key: video_paths[name]
                for name, _, _, key in camera_config(cameras)
            },
        })
    data_path = output / f"data/chunk-000/episode_{episode_index:06d}.parquet"
    data_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pylist(rows, schema=parquet_schema(cameras, mobile=mobile)),
        data_path,
    )
    task = str(manifest.get("task", "check joint states"))
    episode_meta = {
        "episode_index": episode_index, "tasks": [task], "length": len(rows),
        "data_path": str(data_path.relative_to(output)),
        "video_paths": video_paths,
        "raw_episode": str(raw),
        "raw_primary_frames": len(primary_rows),
        "primary_camera": primary,
        "common_interval_primary_frames": len(aligned_primary),
    }
    return rows, episode_meta, missing_count


def convert_dataset(
    raw_root: Path,
    output: Path,
    stride: int,
    overwrite: bool,
    missing_gripper: str,
    cameras: list[str],
    *,
    mobile: bool = False,
) -> None:
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"output exists: {output}; use --overwrite")
        shutil.rmtree(output)
    output.mkdir(parents=True)
    episodes = sorted(raw_root.glob("episodes/episode_*"), key=lambda p: int(p.name.split("_")[-1]))
    if not episodes:
        raise RuntimeError(f"no episodes found under {raw_root / 'episodes'}")

    all_rows, episode_meta, total_missing_gripper = [], [], 0
    for new_index, raw in enumerate(episodes):
        rows, meta, missing_count = convert_episode(
            raw, output, new_index, stride, len(all_rows), missing_gripper,
            cameras, mobile
        )
        all_rows.extend(rows)
        episode_meta.append(meta)
        total_missing_gripper += missing_count
        print(f"{raw.name}: {len(rows)} frames")

    meta_dir = output / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    tasks = sorted({task for item in episode_meta for task in item["tasks"]})
    task_to_index = {task: i for i, task in enumerate(tasks)}
    for item in episode_meta:
        item["task_index"] = task_to_index[item["tasks"][0]]
    (meta_dir / "tasks.jsonl").write_text(
        "".join(json.dumps({"task_index": i, "task": task}) + "\n" for i, task in enumerate(tasks))
    )
    (meta_dir / "episodes.jsonl").write_text(
        "".join(json.dumps(item) + "\n" for item in episode_meta)
    )
    write_stats(all_rows, meta_dir, mobile=mobile)

    image_mapping = {
        key: key for _, _, _, key in camera_config(cameras)
    }
    schema_id = "jaka_v21_mobile" if mobile else "jaka_v21_arm_only"
    canonical_dim = 10 if mobile else 8
    delta = [True] * 6 + [False] * (4 if mobile else 2)
    source_state = ["observation.joints", "observation.gripper"]
    source_state_dims = [6, 2]
    source_action = ["action"]
    source_action_dims = [7]
    if mobile:
        source_state.append("observation.agv")
        source_state_dims.append(9)
        source_action.append("observation.agv")
        source_action_dims.append(9)
    (meta_dir / "labvla_manifest.json").write_text(json.dumps({
        "version": 1, "schema_id": schema_id,
        "robot_type": "jaka+agv" if mobile else "jaka",
        "state": {"keys": ["observation.state"], "dims": [canonical_dim]},
        "action": {"keys": ["action"], "dims": [canonical_dim],
                    "delta": delta,
                    "gripper_dims": [7]},
        "images": image_mapping,
        "gripper_semantic": "open_fraction",
        "source_state": {"keys": source_state, "dims": source_state_dims},
        "source_action": {"keys": source_action, "dims": source_action_dims},
    }, indent=2) + "\n")
    output_fps = 30.0 / stride
    (meta_dir / "info.json").write_text(json.dumps({
        "codebase_version": "v2.1", "schema_id": schema_id,
        "robot_type": "jaka+agv" if mobile else "jaka", "fps": output_fps,
        "total_episodes": len(episode_meta), "total_frames": len(all_rows),
        "total_tasks": len(tasks), "total_chunks": 1,
        "total_videos": len(episode_meta) * len(cameras), "video": True, "encoding": "mp4v",
        "features": {
            "observation.state": {"dtype": "float32", "shape": [22 if mobile else 13]},
            "observation.joints": {"dtype": "float32", "shape": [6]},
            "observation.tcp_pose": {"dtype": "float32", "shape": [6]},
            "observation.gripper": {"dtype": "float32", "shape": [2]},
            "observation.agv": {"dtype": "float32", "shape": [9]},
            "action": {"dtype": "float32", "shape": [7]},
            **{
                key: {"dtype": "video", "shape": [400, 640, 3], "video_ext": "mp4"}
                for _, _, _, key in camera_config(cameras)
            },
        },
        "timestamp_semantics": f"{cameras[0]} camera timestamp",
        "alignment": {
            "state": "linear interpolation at selected primary timestamps",
            "action": "next selected state",
            "secondary_cameras": "nearest camera timestamp",
            "cameras": cameras,
            "downsample_stride": stride,
        },
    }, indent=2) + "\n")
    (meta_dir / "conversion_report.json").write_text(json.dumps({
        "raw_root": str(raw_root.resolve()), "output_root": str(output.resolve()),
        "input_fps": 30, "output_fps": output_fps, "stride": stride,
        "episodes": len(episode_meta), "frames": len(all_rows),
        "gripper_missing_rows": total_missing_gripper,
        "actions_csv_used": False,
        "action_source": (
            "next_selected_state_arm_plus_measured_agv" if mobile
            else "next_selected_state"
        ),
        "cameras": cameras,
        "self_validation": "passed",
    }, indent=2) + "\n")
    print(f"wrote {output} ({len(episode_meta)} episodes, {len(all_rows)} frames, {output_fps:g} Hz)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-root", type=Path, required=True,
        help="Raw dataset root containing episodes/episode_*.",
    )
    parser.add_argument(
        "--output-parent", type=Path, required=True,
        help="Parent directory for the generated LeRobot datasets.",
    )
    parser.add_argument(
        "--missing-gripper", choices=("interpolate", "zero", "error"),
        default="interpolate",
        help="How to handle missing gripper_openness rows (default: timestamp interpolation).",
    )
    parser.add_argument(
        "--only", choices=("30hz", "10hz", "both"), default="both",
        help="Generate only one output rate, or both (default).",
    )
    parser.add_argument(
        "--output-prefix", type=str, default=None,
        help="Output directory prefix; defaults to jaka_mobile_rgb3_lerobot for --mobile, otherwise jaka_rgb3_lerobot.",
    )
    parser.add_argument(
        "--cameras", type=parse_camera_list, default=parse_camera_list("front,side,wrist"),
        help="Comma-separated camera order, 1-3 of front,side,wrist (default: front,side,wrist).",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--mobile", action="store_true",
        help="Preserve observation.agv and emit the jaka_v21_mobile 10-D contract.",
    )
    args = parser.parse_args()
    output_prefix = args.output_prefix or (
        f"jaka_mobile_rgb{len(args.cameras)}_lerobot"
        if args.mobile else f"jaka_rgb{len(args.cameras)}_lerobot"
    )
    if args.only in ("30hz", "both"):
        convert_dataset(
            args.raw_root.resolve(), args.output_parent / f"{output_prefix}_30hz",
            stride=1, overwrite=args.overwrite, missing_gripper=args.missing_gripper,
            cameras=args.cameras, mobile=args.mobile,
        )
    if args.only in ("10hz", "both"):
        convert_dataset(
            args.raw_root.resolve(), args.output_parent / f"{output_prefix}_10hz",
            stride=3, overwrite=args.overwrite, missing_gripper=args.missing_gripper,
            cameras=args.cameras, mobile=args.mobile,
        )


if __name__ == "__main__":
    main()
