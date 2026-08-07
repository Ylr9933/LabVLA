"""JAKA arm-only schemas for raw LeRobot v2.1 datasets."""

from src.schema import DatasetSchema


def _camera_mapping(count: int) -> dict[str, str]:
    return {
        f"observation.images.image{i}": f"observation.images.image{i}"
        for i in range(count)
    }


SCHEMA_RGB3 = DatasetSchema(
    # Keep the historical id: JakaStateGripperTransformFn uses this id to
    # activate the raw-7D -> canonical-8D state/action conversion.
    schema_id="jaka_v21_arm_only",
    robot_type="jaka",
    state_keys=("observation.state",),
    action_keys=("action",),
    state_dims=(8,),
    action_dims=(8,),
    delta_mask=(True, True, True, True, True, True, False, False),
    gripper_action_dims=(7,),
    image_mapping=_camera_mapping(3),
    source="manifest",
    source_path=__file__,
    source_state_keys=("observation.joints", "observation.gripper"),
    source_action_keys=("action",),
    source_state_dims=(6, 2),
    source_action_dims=(7,),
    gripper_semantic="open_fraction",
)

SCHEMA_RGB1 = DatasetSchema(
    schema_id="jaka_v21_arm_only",
    robot_type="jaka",
    state_keys=("observation.state",),
    action_keys=("action",),
    state_dims=(8,), action_dims=(8,),
    delta_mask=(True, True, True, True, True, True, False, False),
    gripper_action_dims=(7,),
    image_mapping=_camera_mapping(1),
    source="manifest", source_path=__file__,
    source_state_keys=("observation.joints", "observation.gripper"),
    source_action_keys=("action",), source_state_dims=(6, 2),
    source_action_dims=(7,), gripper_semantic="open_fraction",
)

SCHEMA_RGB2 = DatasetSchema(
    schema_id="jaka_v21_arm_only",
    robot_type="jaka",
    state_keys=("observation.state",),
    action_keys=("action",),
    state_dims=(8,), action_dims=(8,),
    delta_mask=(True, True, True, True, True, True, False, False),
    gripper_action_dims=(7,),
    image_mapping=_camera_mapping(2),
    source="manifest", source_path=__file__,
    source_state_keys=("observation.joints", "observation.gripper"),
    source_action_keys=("action",), source_state_dims=(6, 2),
    source_action_dims=(7,), gripper_semantic="open_fraction",
)

# Keep the historical one-camera entry available for old checkpoints and
# datasets. The RGB3 training launcher uses SCHEMA_RGB3 explicitly.
SCHEMA = DatasetSchema(
    schema_id="jaka_v21_arm_only",
    robot_type="jaka",
    state_keys=("observation.state",),
    action_keys=("action",),
    state_dims=(8,),
    action_dims=(8,),
    delta_mask=(True, True, True, True, True, True, False, False),
    gripper_action_dims=(7,),
    image_mapping={"observation.images.front": "observation.images.image0"},
    source="manifest",
    source_path=__file__,
    source_state_keys=("observation.joints", "observation.gripper"),
    source_action_keys=("action",),
    source_state_dims=(6, 2),
    source_action_dims=(7,),
    gripper_semantic="open_fraction",
)
