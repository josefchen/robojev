"""Settings for using RoboJev as a LeRobot reward model."""

from __future__ import annotations

from dataclasses import dataclass, field

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature
from lerobot.configs.rewards import RewardModelConfig
from lerobot.utils.constants import OBS_IMAGES


@RewardModelConfig.register_subclass("robojev")
@dataclass
class RoboJevConfig(RewardModelConfig):
    """RoboJev answers "did the robot complete the task?" with a probability.

    RoboJev runs as a server (see the model card at huggingface.co/josefchen/RoboJev-27B), so
    this reward model has no weights of its own: it sends four frames from each attempt (the
    start, a third, two thirds and the end) and the task sentence to that server.

    Args:
        server_url: Where the RoboJev server answers, such as ``http://localhost:8000``.
        served_model: The ``--served-model-name`` the server was started with.
        headers: Extra HTTP headers, for example access keys for a hosted endpoint.
        image_key: Observation key of the camera RoboJev looks at.
        task_key: Key holding the task sentence in the batch.
        default_task: Task sentence to use when the batch has none.
        success_threshold: If set, ``compute_reward`` returns 1.0 at or above this probability
            and 0.0 below it, instead of the probability itself.
        timeout_s: Seconds to wait for one answer.
        workers: Answers requested at the same time.
    """

    pretrained_path: str | None = None

    server_url: str = "http://localhost:8000"
    served_model: str = "robojev"
    headers: dict[str, str] = field(default_factory=dict)

    image_key: str = OBS_IMAGES + ".front"
    task_key: str = "task"
    default_task: str | None = None
    success_threshold: float | None = None
    timeout_s: float = 120.0
    workers: int = 8

    license: str | None = "apache-2.0"
    tags: list[str] | None = field(default_factory=lambda: ["reward-model", "success-detection", "robojev"])

    input_features: dict[str, PolicyFeature] = field(default_factory=dict)
    output_features: dict[str, PolicyFeature] = field(default_factory=dict)
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {"VISUAL": NormalizationMode.IDENTITY, "REWARD": NormalizationMode.IDENTITY}
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.success_threshold is not None and not 0.0 < self.success_threshold < 1.0:
            raise ValueError(f"success_threshold must be between 0 and 1, got {self.success_threshold}")
        if self.workers < 1:
            raise ValueError(f"workers must be at least 1, got {self.workers}")
        if self.image_key not in self.input_features:
            self.input_features[self.image_key] = PolicyFeature(shape=(3, 288, 512), type=FeatureType.VISUAL)
        self.output_features.setdefault("reward", PolicyFeature(shape=(1,), type=FeatureType.REWARD))

    @property
    def observation_delta_indices(self) -> list[int] | None:
        return None

    @property
    def action_delta_indices(self) -> None:
        return None

    @property
    def reward_delta_indices(self) -> None:
        return None

    def validate_features(self) -> None:
        if self.image_key not in self.input_features:
            raise ValueError(f"RoboJev needs the image input feature {self.image_key!r}")
