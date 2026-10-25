"""Optional CPU integration checks: install -e ./lerobot_plugin to run these."""

import pytest

pytest.importorskip("lerobot")
pytest.importorskip("lerobot_reward_robojev")
import torch
from lerobot.rewards import (
    make_reward_model,
    make_reward_model_config,
    make_reward_pre_post_processors,
)


@pytest.mark.parametrize("threshold, expected", [(None, [0.25, 0.9]), (0.8, [0.0, 1.0])])
def test_registered_reward_model_scores_complete_episodes(monkeypatch, threshold, expected):
    config = make_reward_model_config("robojev", device="cpu", success_threshold=threshold)
    model = make_reward_model(config)
    preprocessor, postprocessor = make_reward_pre_post_processors(config)
    assert preprocessor is not None and postprocessor is not None
    seen = []

    def score(tasks, clips, workers):
        seen.append((tasks, clips, workers))
        return [0.25, 0.9]

    monkeypatch.setattr(model.client, "done_many", score)
    frames = torch.arange(2 * 21 * 3 * 8 * 8, dtype=torch.float32).reshape(2, 21, 3, 8, 8)
    rewards = model.compute_reward({config.image_key: frames, "task": ["Task A", "Task B"]})
    assert rewards.tolist() == pytest.approx(expected)
    tasks, clips, _ = seen[0]
    assert tasks == ["Task A", "Task B"]
    assert len(clips) == 2 and all(len(clip) == 10 for clip in clips)
    assert torch.equal(clips[0][0], frames[0, 0])
    assert torch.equal(clips[0][-1], frames[0, -1])
