#!/usr/bin/env python
"""JAKA + AGV deployment entry point."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

# Support direct execution from the repository root without requiring callers
# to pre-populate PYTHONPATH.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from deployment.serve_jaka import _checkpoint_schema, _option_value


EXPECTED_SCHEMA_IDS = {"jaka_v21_mobile", "jaka_v21_mobile_action9"}
EXPECTED_DIM = 10
EXPECTED_MASK = [True, True, True, True, True, True, False, False, False, False]


def validate_jaka_mobile_checkpoint(checkpoint: Path) -> None:
    schema = _checkpoint_schema(checkpoint)
    schema_id = schema.get("schema_id")
    if schema_id not in EXPECTED_SCHEMA_IDS:
        raise SystemExit(
            f"[JAKA mobile] unsupported schema_id={schema_id!r}; "
            f"expected one of {sorted(EXPECTED_SCHEMA_IDS)!r}."
        )
    if sum(schema.get("state_dims", ())) != EXPECTED_DIM:
        raise SystemExit(f"[JAKA mobile] expected state_dim=10, got {schema.get('state_dims')!r}")
    if sum(schema.get("action_dims", ())) != EXPECTED_DIM:
        raise SystemExit(f"[JAKA mobile] expected action_dim=10, got {schema.get('action_dims')!r}")
    if list(schema.get("delta_mask", ())) != EXPECTED_MASK:
        raise SystemExit(f"[JAKA mobile] unexpected delta_mask={schema.get('delta_mask')!r}")
    if list(schema.get("gripper_action_dims", ())) != [7]:
        raise SystemExit(
            f"[JAKA mobile] expected gripper_action_dims=[7], "
            f"got {schema.get('gripper_action_dims')!r}"
        )
    print(
        "[JAKA mobile] checkpoint contract validated: "
        f"schema_id={schema_id}, state_dim=10, action_dim=10"
    )


def main() -> None:
    argv = sys.argv[1:]
    checkpoint_arg = _option_value(argv, "pretrained_path")
    if not checkpoint_arg:
        raise SystemExit("[JAKA mobile] --pretrained_path is required")
    checkpoint = Path(checkpoint_arg).expanduser().resolve()
    if not checkpoint.is_dir():
        raise SystemExit(f"[JAKA mobile] checkpoint directory does not exist: {checkpoint}")
    validate_jaka_mobile_checkpoint(checkpoint)

    action_dim = _option_value(argv, "action_dim")
    if action_dim is not None and int(action_dim) != EXPECTED_DIM:
        raise SystemExit(f"[JAKA mobile] --action_dim must be 10, got {action_dim!r}")
    robot_type = _option_value(argv, "robot_type")
    if robot_type is not None and robot_type != "jaka+agv":
        raise SystemExit(f"[JAKA mobile] --robot_type must be 'jaka+agv', got {robot_type!r}")

    forwarded = list(argv)
    if action_dim is None:
        forwarded.extend(["--action_dim", str(EXPECTED_DIM)])
    if robot_type is None:
        forwarded.extend(["--robot_type", "jaka+agv"])
    target = Path(__file__).with_name("serve_labvla.py").resolve()
    sys.argv = [str(target), *forwarded]
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()
