from __future__ import annotations

import argparse
import logging
from typing import Any

import numpy as np
import torch
from PIL import Image

from serving import WebsocketPolicyServer
from xvla_client.base_policy import BasePolicy


def to_image(value: Any) -> Image.Image:
    image = np.asarray(value)
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return Image.fromarray(image)


class XVLAWebsocketPolicy(BasePolicy):
    def __init__(self, model_path: str, device: str, default_prompt: str | None) -> None:
        from datasets.domain_handler import slai_piper_space
        from models.modeling_xvla import XVLA
        from models.processing_xvla import XVLAProcessor

        self.model_path = model_path
        self.processor = XVLAProcessor.from_pretrained(model_path)
        self.model = XVLA.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.float32)
        self.device = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else "cpu" if device == "auto" else device)
        self.model = self.model.to(self.device).to(torch.float32).eval()
        self.action_mode = self.model.config.action_mode.lower()
        self.default_prompt = default_prompt
        self.slai_piper_space = slai_piper_space

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "backend": "xvla",
            "model_path": self.model_path,
            "ckpt_dir": self.model_path,
            "checkpoint_dir": self.model_path,
            "action_mode": self.action_mode,
            "action_dim": self.model.action_space.dim_action,
            "action_horizon": self.model.num_actions,
        }

    def proprio(self, obs: dict[str, Any]) -> np.ndarray:
        state = np.asarray(obs.get("observation.state", obs.get("proprio")), dtype=np.float32)
        dim = getattr(self.model.action_space, "dim_proprio", self.model.action_space.dim_action)
        if state.shape[-1] == dim:
            return state
        return self.slai_piper_space.extract_state_action(state, None, self.action_mode, self.action_mode)["state"]

    @torch.no_grad()
    def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        keys = ("observation.images.cam_high", "observation.images.cam_left_wrist", "observation.images.cam_right_wrist")
        images = [to_image(obs[key]) for key in keys if key in obs]
        prompt = obs.get("prompt") or obs.get("language_instruction") or self.default_prompt
        if not images or prompt is None:
            raise ValueError("X-VLA websocket policy requires images and a prompt")
        inputs = self.processor(images, prompt)
        inputs = {key: value.to(self.device, dtype=torch.float32) if value.is_floating_point() else value.to(self.device) for key, value in inputs.items()}
        actions = self.model.generate_actions(
            **inputs,
            proprio=torch.as_tensor(self.proprio(obs), dtype=torch.float32, device=self.device).unsqueeze(0),
            domain_id=torch.tensor([int(obs.get("domain_id", 19))], dtype=torch.long, device=self.device),
            steps=int(obs.get("steps", 10)),
        )[0].float().cpu().numpy()
        return {"actions": actions}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve an X-VLA policy over the xvla-client websocket protocol.")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--default_prompt", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, force=True)
    policy = XVLAWebsocketPolicy(args.model_path, args.device, args.default_prompt)
    server = WebsocketPolicyServer(policy, host=args.host, port=args.port, metadata=policy.metadata)
    server.serve_forever()


if __name__ == "__main__":
    main()
