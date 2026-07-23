"""JAKA 6-DoF arm-only schema for the raw LeRobot v2.1 dataset."""

from src.schema import DatasetSchema


SCHEMA = DatasetSchema(
    schema_id="jaka_v21_arm_only",
    robot_type="jaka",
    state_keys=("observation.state",),
    action_keys=("action",),
    state_dims=(8,),
    action_dims=(8,),
    delta_mask=(True, True, True, True, True, True, False, False),
    gripper_action_dims=(7,),
    action_loss_mask=(True, True, True, True, True, True, False, True),
    image_mapping={"observation.images.front": "observation.images.image0"},
    source="manifest",
    source_path=__file__,
    source_state_keys=("observation.joints", "observation.gripper"),
    source_action_keys=("action",),
    source_state_dims=(6, 2),
    source_action_dims=(7,),
    gripper_semantic="open_fraction",
)
