"""Processors for the RoboJev reward model, and a reward step for real-robot learning.

``make_robojev_pre_post_processors`` is what LeRobot's reward factory looks up. Both pipelines
pass data through: frames are sent to the server as they come.

``RoboJevRewardProcessorStep`` goes in an environment pipeline (for example LeRobot's
real-robot RL loop, in place of the reward classifier step). It keeps the episode's frames from
one camera and, every few steps, asks RoboJev whether the task is done yet, using ten frames evenly
spaced from the start to the latest frame. When the answer reaches the threshold, it gives the
success reward and, if asked to, ends the episode.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.lerobot_types import EnvTransition, TransitionKey
from lerobot.processor import (
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
    policy_action_to_transition,
)
from lerobot.utils.constants import OBS_IMAGES, POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME

from .client import RoboJevClient, compact, pick_frames
from .configuration_robojev import RoboJevConfig


def make_robojev_pre_post_processors(
    config: RoboJevConfig,
    dataset_stats: dict[str, dict[str, Any]] | None = None,  # noqa: ARG001 (no statistics are used)
) -> tuple[PolicyProcessorPipeline[dict[str, Any], dict[str, Any]], PolicyProcessorPipeline[PolicyAction, PolicyAction]]:
    preprocessor = PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](steps=[], name=POLICY_PREPROCESSOR_DEFAULT_NAME)
    postprocessor = PolicyProcessorPipeline(name=POLICY_POSTPROCESSOR_DEFAULT_NAME, to_transition=policy_action_to_transition)
    return preprocessor, postprocessor


@dataclass
@ProcessorStepRegistry.register("robojev_reward_processor")
class RoboJevRewardProcessorStep(ProcessorStep):
    """Rewards an episode when RoboJev says its task is done.

    Attributes:
        task: The task sentence, such as "Put the ring on the peg".
        server_url: Where the RoboJev server answers.
        served_model: The server's ``--served-model-name``.
        headers: Extra HTTP headers, for example access keys for a hosted endpoint.
        image_key: Observation key of the camera to judge from.
        check_every: Ask RoboJev every this many steps (each answer takes about half a second).
        min_steps: Do not ask before the episode has this many steps.
        success_threshold: Probability at which the task counts as done.
        success_reward: Reward given when it does.
        terminate_on_success: End the episode when the task is done.
    """

    task: str = ""
    server_url: str = "http://localhost:8000"
    served_model: str = "robojev"
    headers: dict[str, str] = field(default_factory=dict)
    image_key: str = OBS_IMAGES + ".front"
    check_every: int = 10
    min_steps: int = 10
    success_threshold: float = 0.8
    success_reward: float = 1.0
    terminate_on_success: bool = True

    def __post_init__(self) -> None:
        if not self.task:
            raise ValueError("RoboJevRewardProcessorStep needs the task sentence")
        self.client = RoboJevClient(self.server_url, self.served_model, headers=self.headers)
        self.frames: list[bytes] = []  # JPEG at RoboJev's size: about 30 KB a frame
        self.last_probability: float | None = None

    def reset(self) -> None:
        self.frames = []
        self.last_probability = None

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        new_transition = transition.copy()
        observation = new_transition.get(TransitionKey.OBSERVATION) or {}
        if self.image_key not in observation:
            return new_transition
        frame = observation[self.image_key]
        frame = frame[0] if getattr(frame, "ndim", 0) == 4 else frame  # drop a batch dimension of one
        self.frames.append(compact(frame))
        steps = len(self.frames)
        info = dict(new_transition.get(TransitionKey.INFO) or {})
        if steps >= self.min_steps and (steps - self.min_steps) % self.check_every == 0:
            started = time.perf_counter()
            self.last_probability = self.client.done(self.task, [self.frames[i] for i in pick_frames(steps)])
            info["robojev_seconds"] = time.perf_counter() - started
            if self.last_probability >= self.success_threshold:
                new_transition[TransitionKey.REWARD] = self.success_reward
                if self.terminate_on_success:
                    new_transition[TransitionKey.DONE] = True
        if self.last_probability is not None:
            info["robojev_p_done"] = self.last_probability
        new_transition[TransitionKey.INFO] = info
        return new_transition

    def get_config(self) -> dict[str, Any]:
        return {"task": self.task, "server_url": self.server_url, "served_model": self.served_model,
                "image_key": self.image_key, "check_every": self.check_every, "min_steps": self.min_steps,
                "success_threshold": self.success_threshold, "success_reward": self.success_reward,
                "terminate_on_success": self.terminate_on_success}

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features
