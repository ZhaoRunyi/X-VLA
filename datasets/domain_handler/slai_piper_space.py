from __future__ import annotations

import json
from pathlib import Path

from . import slai_piper_policy as policy

PREFIX = "slai_piper_"
DEFAULT_SPACE = "ee_gripper"
GRIPPER_THRESHOLD_PATH = Path(__file__).with_name("slai_piper_gripper_thresholds.json")

def split_space_name(name: str | None) -> str:
    if not name or name == "slai_piper":
        return DEFAULT_SPACE
    if name.startswith(PREFIX):
        return name[len(PREFIX):]
    raise ValueError(f"SLAI Piper spaces must be named '{PREFIX}<policy_space>', got {name!r}")

def make_space_configs(state_name: str | None, action_name: str | None = None):
    state_ids = split_space_name(state_name)
    action_ids = split_space_name(action_name or state_name)
    return policy.StateSpaceConfig(ids=state_ids), policy.ActionSpaceConfig(ids=action_ids)

def extract_state_action(full_state, full_action, state_name: str | None, action_name: str | None = None):
    state_config, action_config = make_space_configs(state_name, action_name)
    return policy.extract_state_action_inputs(full_state, full_action, state_space=state_config, action_space=action_config)

def action_dim(name: str) -> int:
    return policy.get_space_dim(policy.ActionSpaceConfig(ids=split_space_name(name)))

def gripper_indices(name: str) -> tuple[int, ...]:
    names = policy.get_vector_names(policy.ActionSpaceConfig(ids=split_space_name(name)))
    return tuple(index for index, value in enumerate(names) if "gripper" in value)

def gripper_threshold(root: str | Path) -> float:
    root = Path(root)
    thresholds = json.loads(GRIPPER_THRESHOLD_PATH.read_text()) if GRIPPER_THRESHOLD_PATH.exists() else {}
    threshold = thresholds.get(root.name)
    if threshold is None:
        from scripts.estimate_piper_gripper_threshold import estimate_threshold, write_threshold
        threshold = estimate_threshold(root)
        write_threshold(root, threshold)
    return float(threshold)
