#!/usr/bin/env python
"""JAKA deployment entry point.

This module validates the JAKA checkpoint contract, then delegates the actual
WebSocket serving and inference implementation to ``serve_labvla.py``. Keeping
the protocol implementation in one place prevents JAKA and generic LabVLA
deployment from drifting apart.
"""

from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path


EXPECTED_SCHEMA_ID = "jaka_v21_arm_only"
EXPECTED_DIM = 8


def _checkpoint_schema(checkpoint: Path) -> dict:
    candidates = (
        checkpoint / "labvla_schema.json",
        checkpoint.parent / "labvla_schema.json",
    )
    for path in candidates:
        if path.is_file():
            try:
                with path.open() as f:
                    schema = json.load(f)
            except (OSError, json.JSONDecodeError) as exc:
                raise SystemExit(f"[JAKA] invalid schema file {path}: {exc}") from exc
            if not isinstance(schema, dict):
                raise SystemExit(f"[JAKA] schema must be a JSON object: {path}")
            return schema
    raise SystemExit(
        "[JAKA] checkpoint has no labvla_schema.json. "
        "Use a checkpoint produced by the current LabVLA training pipeline."
    )


def validate_jaka_checkpoint(checkpoint: Path) -> None:
    schema = _checkpoint_schema(checkpoint)
    schema_id = schema.get("schema_id")
    if schema_id != EXPECTED_SCHEMA_ID:
        raise SystemExit(
            f"[JAKA] schema_id={schema_id!r} is not {EXPECTED_SCHEMA_ID!r}."
        )
    for field in ("state_dims", "action_dims", "delta_mask", "gripper_action_dims"):
        if field not in schema:
            raise SystemExit(f"[JAKA] schema is missing required field {field!r}")
    if sum(schema["state_dims"]) != EXPECTED_DIM:
        raise SystemExit(f"[JAKA] expected state_dim=8, got {schema['state_dims']!r}")
    if sum(schema["action_dims"]) != EXPECTED_DIM:
        raise SystemExit(f"[JAKA] expected action_dim=8, got {schema['action_dims']!r}")
    if list(schema["delta_mask"]) != [True, True, True, True, True, True, False, False]:
        raise SystemExit(f"[JAKA] unexpected delta_mask={schema['delta_mask']!r}")
    if list(schema["gripper_action_dims"]) != [7]:
        raise SystemExit(
            f"[JAKA] expected gripper_action_dims=[7], "
            f"got {schema['gripper_action_dims']!r}"
        )
    print(
        "[JAKA] checkpoint contract validated: "
        f"schema_id={schema_id}, state_dim=8, action_dim=8"
    )


def _option_value(argv: list[str], name: str) -> str | None:
    prefix = f"--{name}="
    for index, value in enumerate(argv):
        if value.startswith(prefix):
            return value[len(prefix):]
        if value == f"--{name}" and index + 1 < len(argv):
            return argv[index + 1]
    return None


def main() -> None:
    argv = sys.argv[1:]
    checkpoint_arg = _option_value(argv, "pretrained_path")
    if not checkpoint_arg:
        raise SystemExit("[JAKA] --pretrained_path is required")
    checkpoint = Path(checkpoint_arg).expanduser().resolve()
    if not checkpoint.is_dir():
        raise SystemExit(f"[JAKA] checkpoint directory does not exist: {checkpoint}")
    validate_jaka_checkpoint(checkpoint)

    action_dim = _option_value(argv, "action_dim")
    if action_dim is not None and int(action_dim) != EXPECTED_DIM:
        raise SystemExit(f"[JAKA] --action_dim must be 8, got {action_dim!r}")
    robot_type = _option_value(argv, "robot_type")
    if robot_type is not None and robot_type != "jaka":
        raise SystemExit(f"[JAKA] --robot_type must be 'jaka', got {robot_type!r}")

    forwarded = list(argv)
    if action_dim is None:
        forwarded.extend(["--action_dim", str(EXPECTED_DIM)])
    if robot_type is None:
        forwarded.extend(["--robot_type", "jaka"])

    target = Path(__file__).with_name("serve_labvla.py").resolve()
    sys.argv = [str(target), *forwarded]
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()
