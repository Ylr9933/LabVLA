"""JAKA arm + AGV velocity schema for the raw LeRobot v2.1 dataset."""

from src.schema import DatasetSchema


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
