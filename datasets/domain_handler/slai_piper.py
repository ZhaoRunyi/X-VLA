from __future__ import annotations

import random

import numpy as np
import torch
from mmengine import fileio
from PIL import Image
from scipy.interpolate import interp1d

from .base import DomainHandler
from . import slai_piper_policy
from .slai_piper_space import extract_state_action, gripper_indices, gripper_threshold
from ..utils import read_parquet, read_video_to_frames

class SLAIPiperLeRobotV21Handler(DomainHandler):
    CAMERA_VIEW = tuple(slai_piper_policy.get_image_key_map(slai_piper_policy.ImageSpaceConfig()).values())

    def _episode_paths(self, item: dict) -> tuple[str, int]:
        episode_index = item["episode_index"]
        episode_chunk = episode_index // self.meta["chunks_size"]
        data_path = fileio.join_path(self.meta["root_path"], self.meta["data_path"]).format(
            episode_chunk=episode_chunk, episode_index=episode_index
        )
        return data_path, episode_chunk

    def _read_images(self, episode_index: int, episode_chunk: int) -> list[np.ndarray]:
        if not self.meta.get("video_path"):
            return []
        images = []
        for video_key in self.CAMERA_VIEW[: self.num_views]:
            video_path = fileio.join_path(self.meta["root_path"], self.meta["video_path"]).format(
                episode_chunk=episode_chunk, episode_index=episode_index, video_key=video_key
            )
            images.append(read_video_to_frames(video_path))
        return images

    def iter_episode(self, traj_idx: int, *, num_actions: int, training: bool, image_aug,
                     lang_aug_map: dict | None, action_mode: str, **kwargs):
        item = self.meta["datalist"][traj_idx]
        data_path, episode_chunk = self._episode_paths(item)
        data = read_parquet(data_path)
        images = self._read_images(item["episode_index"], episode_chunk)
        selected = extract_state_action(data["observation.state"], data["action"], action_mode, action_mode)
        states = np.asarray(selected["state"], dtype=np.float32)
        actions = np.asarray(selected["actions"], dtype=np.float32)
        if "slai_piper" in action_mode:
            indices = gripper_indices(action_mode)
            if indices:
                threshold = gripper_threshold(self.meta["root_path"])
                states[..., indices], actions[..., indices] = states[..., indices] >= threshold, actions[..., indices] >= threshold

        freq = float(self.meta.get("fps", 10))
        duration = 1.0
        t = np.arange(actions.shape[0], dtype=np.float64) / freq
        action_fn = interp1d(t, actions, axis=0, bounds_error=False, fill_value=(actions[0], actions[-1]))
        horizon = max(1, int(round(freq * duration)))
        idxs = list(range(0, max(1, actions.shape[0] - horizon)))
        if training:
            random.shuffle(idxs)

        instruction = item["tasks"][0]
        if not isinstance(instruction, str):
            instruction = str(instruction)
        image_mask = torch.zeros(self.num_views, dtype=torch.bool)
        image_mask[: min(len(images), self.num_views)] = True
        for idx in idxs:
            image_input = self._build_image_tensor(images, idx, image_aug)
            q = np.linspace(t[idx], min(t[idx] + duration, float(t.max())), num_actions + 1, dtype=np.float32)
            action = torch.tensor(action_fn(q)[1:], dtype=torch.float32)
            if action.numel() and (action[0] - action[-1]).abs().max() < 1e-5:
                continue
            if training and lang_aug_map and instruction in lang_aug_map:
                instruction = random.choice(lang_aug_map[instruction])
            yield {
                "language_instruction": instruction,
                "image_input": image_input,
                "image_mask": image_mask,
                "proprio": torch.tensor(states[idx], dtype=torch.float32),
                "action": action,
            }

    def _build_image_tensor(self, images: list[np.ndarray], idx: int, image_aug) -> torch.Tensor:
        frames = []
        for view in range(min(self.num_views, len(images))):
            frame_idx = min(idx, len(images[view]) - 1)
            frames.append(image_aug(Image.fromarray(images[view][frame_idx])))
        if not frames:
            raise ValueError("SLAI Piper LeRobot data requires at least one image stream.")
        while len(frames) < self.num_views:
            frames.append(torch.zeros_like(frames[0]))
        return torch.stack(frames, 0)
