"""Score a complete video through LeRobot's reward factory and your RoboJev vLLM server.

Install the client and plugin from the repository root:
    python -m pip install -e . -e ./lerobot_plugin

Then, with ffmpeg on PATH and your model server running:
    python examples/score_lerobot_episode.py "Put the ring on the peg" attempt.mp4 --server http://localhost:8000

The LeRobot process runs on CPU. The remote vLLM server hosts the model on a GPU.
"""

import argparse

# Register the out-of-tree reward model before using the factory.
import lerobot_reward_robojev  # noqa: F401
import numpy as np
import torch
from lerobot.rewards import make_reward_model, make_reward_model_config

from robojev import video_frames


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task")
    parser.add_argument("video")
    parser.add_argument("--server", default="http://localhost:8000")
    args = parser.parse_args()

    frames = video_frames(args.video, 10)
    clip = torch.stack([torch.from_numpy(np.array(frame)).permute(2, 0, 1) for frame in frames])
    config = make_reward_model_config("robojev", server_url=args.server, device="cpu")
    reward_model = make_reward_model(config)
    with torch.inference_mode():
        score = reward_model.compute_reward({"observation.images.front": clip.unsqueeze(0), "task": [args.task]})
    print(f"P(task completed): {score[0].item():.3f}")


if __name__ == "__main__":
    main()
