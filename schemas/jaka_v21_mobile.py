"""JAKA arm + AGV velocity schema for the raw LeRobot v2.1 dataset."""

from src.schema import DatasetSchema


def _camera_mapping(count: int) -> dict[str, str]:
    return {
        f"observation.images.image{i}": f"observation.images.image{i}"
        for i in range(count)
    }


SCHEMA = DatasetSchema(
    schema_id="jaka_v21_mobile",
    robot_type="jaka+agv",
    state_keys=("observation.state",),
    action_keys=("action",),
    state_dims=(10,),
    action_dims=(10,),
    # Arm joint targets are delta; reserved slot, gripper, and AGV velocities
    # are absolute targets.
    delta_mask=(True, True, True, True, True, True, False, False, False, False),
    gripper_action_dims=(7,),
    image_mapping={"observation.images.front": "observation.images.image0"},
    source="manifest",
    source_path=__file__,
    source_state_keys=("observation.joints", "observation.gripper", "observation.agv"),
    source_action_keys=("action", "observation.agv"),
    source_state_dims=(6, 2, 9),
    source_action_dims=(7, 9),
    gripper_semantic="open_fraction",
)

# Three-camera variant used by convert_jaka_rgb3_to_lerobot.py --mobile.
# The schema id stays identical so the mobile state/action transform is reused.
SCHEMA_RGB3 = DatasetSchema(
    schema_id="jaka_v21_mobile",
    robot_type="jaka+agv",
    state_keys=("observation.state",),
    action_keys=("action",),
    state_dims=(10,),
    action_dims=(10,),
    delta_mask=(True, True, True, True, True, True, False, False, False, False),
    gripper_action_dims=(7,),
    image_mapping=_camera_mapping(3),
    source="manifest",
    source_path=__file__,
    source_state_keys=("observation.joints", "observation.gripper", "observation.agv"),
    source_action_keys=("action", "observation.agv"),
    source_state_dims=(6, 2, 9),
    source_action_dims=(7, 9),
    gripper_semantic="open_fraction",
)


def _rgb_schema(count: int) -> DatasetSchema:
    return DatasetSchema(
        schema_id="jaka_v21_mobile",
        robot_type="jaka+agv",
        state_keys=("observation.state",),
        action_keys=("action",),
        state_dims=(10,), action_dims=(10,),
        delta_mask=(True, True, True, True, True, True, False, False, False, False),
        gripper_action_dims=(7,),
        image_mapping=_camera_mapping(count),
        source="manifest", source_path=__file__,
        source_state_keys=("observation.joints", "observation.gripper", "observation.agv"),
        source_action_keys=("action", "observation.agv"),
        source_state_dims=(6, 2, 9), source_action_dims=(7, 9),
        gripper_semantic="open_fraction",
    )


SCHEMA_RGB1 = _rgb_schema(1)
SCHEMA_RGB2 = _rgb_schema(2)
