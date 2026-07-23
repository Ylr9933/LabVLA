"""JAKA + AGV schema when the action column is already 9-D."""

from src.schema import DatasetSchema


SCHEMA = DatasetSchema(
    schema_id="jaka_v21_mobile_action9",
    robot_type="jaka+agv",
    state_keys=("observation.state",),
    action_keys=("action",),
    state_dims=(10,),
    action_dims=(10,),
    delta_mask=(True, True, True, True, True, True, False, False, False, False),
    gripper_action_dims=(7,),
    image_mapping={"observation.images.front": "observation.images.image0"},
    source="manifest",
    source_path=__file__,
    source_state_keys=("observation.joints", "observation.gripper", "observation.agv"),
    source_action_keys=("action",),
    source_state_dims=(6, 2, 9),
    source_action_dims=(9,),
    gripper_semantic="open_fraction",
)
