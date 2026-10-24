# lerobot-reward-robojev

The package integrates LeRobot with [RoboJev-27B](https://huggingface.co/josefchen/RoboJev-27B), which predicts the likelihood of a robot successfully completing a task from 10 frames of the attempt.

The package includes the reward model (`type: robojev`), which applies to recorded episodes, and a pipeline step for training on the real robot. Both use a RoboJev server over HTTP, hence no weights are loaded in the LeRobot process.

```bash
vllm serve josefchen/RoboJev-27B --served-model-name robojev --max-model-len 32768 \
  --limit-mm-per-prompt '{"image": 16}' --logprobs-mode processed_logprobs --trust-request-chat-template
hf download josefchen/RoboJev-27B --include "lerobot_plugin/*" --local-dir robojev
pip install ./robojev/lerobot_plugin
```

LeRobot will only auto-import plugins that start with some prefixes, and reward models don't have any of those. So you will need to do `import lerobot_reward_robojev` before building either of them.

## Reward step for real-robot training

In an environment pipeline, `RewardClassifierProcessorStep` can be replaced by `RoboJevRewardProcessorStep`. The processor will save the frames of the episode of one camera and query RoboJev every `check_every` steps with ten frames evenly distributed across the episode so far, from the first frame to the most recent. Once P(completed) reaches `success_threshold`, the reward is set to `success_reward`, and, if `terminate_on_success` is true, the episode is terminated. The task is presented as a sentence so no classifier needs to be trained per task.

```python
import lerobot_reward_robojev
from lerobot_reward_robojev import RoboJevRewardProcessorStep

step = RoboJevRewardProcessorStep(
    task="Put the ring on the peg",
    server_url="http://gpu-host:8000",
    image_key="observation.images.front",
    check_every=10,
    success_threshold=0.8,
)
```

The step adds a `robojev_p_done` entry to the `info` of each transition. Each query takes about 0.5 s.

## Reward model for recorded episodes

```python
import lerobot_reward_robojev
from lerobot.rewards import make_reward_model, make_reward_model_config

config = make_reward_model_config("robojev", server_url="http://gpu-host:8000",
                                  image_key="observation.images.front")
reward = make_reward_model(config)
p_done = reward.compute_reward({"observation.images.front": frames,   # (batch, time, 3, H, W)
                                "task": tasks})                         # one sentence per episode
```

`compute_reward` will sample 10 frames uniformly from the first frame through the last frame for each episode. It returns one probability per episode or 1.0 or 0.0 if `success_threshold` is specified.

## Tests

We ran it against the released model on two ArmnetBench ring attempts, with frames resized to 426x240. Our reward model returned 0.989 and 0.033, compared to 0.996 and 0.021 in the benchmark. On the successful attempt, our reward model step returned <0.03 until the last frame, after which it outputted 0.989, set the reward, and terminated the episode.
