"""RoboJev as a LeRobot reward model: one success probability per attempt.

RoboJev has no weights inside this process. ``compute_reward`` picks ten evenly spaced frames from each
attempt in the batch, sends them with the task sentence to a RoboJev server and returns the
probability that the task was completed (or 1.0 / 0.0 against ``success_threshold``).

The batch holds, for each sample, the attempt's frames under ``config.image_key`` as
``(time, channels, height, width)`` (a batch is then ``(batch, time, channels, height, width)``),
and the task sentence under ``config.task_key``. Like TOPReward, it is for inference only: a
saved checkpoint is just ``config.json``.
"""

from __future__ import annotations

import builtins
import os
from pathlib import Path
from typing import Any, TypeVar

import torch
from huggingface_hub import hf_hub_download
from huggingface_hub.constants import CONFIG_NAME
from torch import Tensor

from lerobot.configs.rewards import RewardModelConfig
from lerobot.rewards.pretrained import PreTrainedRewardModel

from .client import RoboJevClient, pick_frames
from .configuration_robojev import RoboJevConfig

T = TypeVar("T", bound="RoboJevRewardModel")


class RoboJevRewardModel(PreTrainedRewardModel):
    """Calls a RoboJev server for each attempt in the batch."""

    config: RoboJevConfig
    name = "robojev"
    config_class = RoboJevConfig

    def __init__(self, config: RoboJevConfig, **kwargs: Any) -> None:
        super().__init__(config)
        self.config = config
        self.client = RoboJevClient(config.server_url, config.served_model, config.timeout_s, config.headers)

    def _tasks(self, batch: dict[str, Any], size: int) -> list[str]:
        tasks = batch.get(self.config.task_key, self.config.default_task)
        if tasks is None:
            raise KeyError(f"No task sentence under {self.config.task_key!r} and no default_task set")
        if isinstance(tasks, str):
            return [tasks] * size
        if len(tasks) != size:
            raise ValueError(f"{len(tasks)} task sentences for {size} attempts")
        return [str(t) for t in tasks]

    def compute_reward(self, batch: dict[str, Any]) -> Tensor:
        """One reward per attempt in the batch."""
        if self.config.image_key not in batch:
            raise KeyError(f"RoboJev batch is missing frames under {self.config.image_key!r}")
        frames = batch[self.config.image_key]
        if frames.ndim == 4:  # one attempt without a batch dimension
            frames = frames.unsqueeze(0)
        if frames.ndim != 5:
            raise ValueError(f"Expected frames as (batch, time, channels, height, width), got shape {tuple(frames.shape)}")
        clips = [[attempt[i] for i in pick_frames(attempt.shape[0])] for attempt in frames]
        probabilities = self.client.done_many(self._tasks(batch, len(clips)), clips, self.config.workers)
        rewards = torch.tensor(probabilities, dtype=torch.float32)
        if self.config.success_threshold is not None:
            rewards = (rewards >= self.config.success_threshold).float()
        return rewards

    def _save_pretrained(self, save_directory: Path) -> None:
        """Save config.json only: the model itself lives behind the server."""
        self.config._save_pretrained(save_directory)

    @classmethod
    def from_pretrained(
        cls: builtins.type[T],
        pretrained_name_or_path: str | Path,
        *,
        config: RewardModelConfig | None = None,
        force_download: bool = False,
        resume_download: bool | None = None,
        proxies: dict | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        strict: bool = False,  # noqa: ARG003 (kept for the shared signature; there are no weights)
        **kwargs: Any,
    ) -> T:
        if config is None:
            if not os.path.isdir(str(pretrained_name_or_path)):
                hf_hub_download(repo_id=str(pretrained_name_or_path), filename=CONFIG_NAME, revision=revision,
                                cache_dir=cache_dir, force_download=force_download, proxies=proxies,
                                token=token, local_files_only=local_files_only)
            config = RewardModelConfig.from_pretrained(
                pretrained_name_or_path=pretrained_name_or_path, force_download=force_download,
                resume_download=resume_download, proxies=proxies, token=token, cache_dir=cache_dir,
                local_files_only=local_files_only, revision=revision, **kwargs)
        if not isinstance(config, RoboJevConfig):
            raise TypeError(f"Expected a RoboJevConfig, got {type(config).__name__}")
        instance = cls(config)
        instance.eval()
        return instance
