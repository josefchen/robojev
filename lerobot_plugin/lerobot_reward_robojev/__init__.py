"""RoboJev for LeRobot: a reward model and a reward step that ask whether a task was completed.

Importing this package registers the ``robojev`` reward model type and the
``robojev_reward_processor`` step with LeRobot.
"""

from .client import RoboJevClient
from .configuration_robojev import RoboJevConfig
from .modeling_robojev import RoboJevRewardModel
from .processor_robojev import RoboJevRewardProcessorStep, make_robojev_pre_post_processors

__all__ = ["RoboJevClient", "RoboJevConfig", "RoboJevRewardModel", "RoboJevRewardProcessorStep",
           "make_robojev_pre_post_processors"]
