# lerobot-reward-robojev

Score a completed robot episode with [RoboJev-27B](https://huggingface.co/josefchen/RoboJev-27B) through LeRobot's reward-model factory. The adapter sends ten evenly spaced frames and a task sentence to your vLLM server and returns one P(task completed) per episode. It does not load model weights in the LeRobot process.

## Install and serve

The plugin requires Python 3.12+ and LeRobot 0.6.x (tested with the published 0.6.1 and current main). The standalone `robojev` HTTP client also supports Python 3.10+. From the repository root:

```bash
python -m pip install -e . -e ./lerobot_plugin
```

On a GPU host with [vLLM installed](https://docs.vllm.ai/en/latest/getting_started/installation/):

```bash
vllm serve josefchen/RoboJev-27B \
  --revision e089a9b69a365744b80765057d86583647af95a9 \
  --served-model-name robojev --max-model-len 32768 \
  --limit-mm-per-prompt '{"image": 16}' --logprobs-mode processed_logprobs --trust-request-chat-template
```

BF16 weights are about 54.7 GB; the released serving setup uses an 80 GB GPU. The client and plugin can run on a separate CPU machine. `server_url` must be your vLLM base URL, not the browser demo URL. There is an [account-free browser demo](https://robo-jev.com/demo) if you only want to try a clip.

Import `lerobot_reward_robojev` before calling the factory: current released LeRobot does not auto-discover reward-plugin packages. The import registers the config, model and processor locally.

## Score a recorded episode

A runnable example is included at [`examples/score_lerobot_episode.py`](../examples/score_lerobot_episode.py). It decodes a complete video with ffmpeg and calls the current LeRobot API:

```bash
python examples/score_lerobot_episode.py "Insert the colourful ring into the central wooden peg" \
  ring-success.mp4 --server http://gpu-host:8000
```

See the [main README](../README.md#use-it) for a pinned download of that released example clip.

To use frames already in memory:

```python
import lerobot_reward_robojev  # registers the out-of-tree reward model
from lerobot.rewards import make_reward_model, make_reward_model_config

config = make_reward_model_config(
    "robojev", server_url="http://gpu-host:8000", device="cpu",
    image_key="observation.images.front",
)
reward = make_reward_model(config)
p_done = reward.compute_reward({
    "observation.images.front": frames,  # torch tensor (batch, time, 3, H, W)
    "task": tasks,                      # one task sentence per episode
})
```

`frames` must contain complete episodes, from start to end. The adapter selects ten evenly spaced frames, including the first and last, from each episode. Images may be uint8 in [0, 255] or floating point in [0, 1]. The result is a CPU float tensor of shape `(batch,)`. Set `success_threshold` to return 1.0 or 0.0 instead of probabilities. Choose a threshold using labelled examples from your own task and camera setup; calibration can change on unseen domains.

## Experimental processor for attempt prefixes

`RoboJevRewardProcessorStep` keeps frames of the current attempt and queries the same server every `check_every` steps after `min_steps`. Call `reset()` at each episode boundary. It adds `robojev_p_done` to the transition's `info` after an answer and records HTTP query time as `robojev_seconds`.

```python
from lerobot_reward_robojev import RoboJevRewardProcessorStep

step = RoboJevRewardProcessorStep(
    task="Put the ring on the peg",
    server_url="http://gpu-host:8000",
    image_key="observation.images.front",
    check_every=10,
    success_threshold=0.8,
    terminate_on_success=False,
)
```

Queries are synchronous and can block an environment step. The prefix processor has not been validated as a control or safety mechanism, and no policy-learning improvement has been demonstrated. A ring can briefly land and then fall off: crossing a threshold early is not proof of a successful completed episode. The processor's class default is `terminate_on_success=True`; the example disables that behavior. Whole-episode scoring is the supported starting point.

## Verification

```bash
python -m pip install -e '.[test]' -e ./lerobot_plugin
pytest -q tests/test_lerobot_integration.py
```

The integration tests use real LeRobot factories on CPU and mocked server probabilities. They check registration, ten-frame sampling, batch output and thresholding without downloading weights. The HTTP transport and ffmpeg paths are covered by the main client's tests. These checks do not measure the model's accuracy or policy-learning performance.
