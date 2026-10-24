import importlib.util
import json
from pathlib import Path

import pytest
from PIL import Image

import robojev
from robojev import robojev as client
from robojev import robojev_overlay as overlay

ROOT = Path(__file__).resolve().parents[1]
DECISION = json.loads((Path(robojev.__file__).parent / "decision.json").read_text())


def test_overlay_picks_the_first_and_latest_frame():
    picked = overlay.pick(100, 10)
    assert len(picked) == 10 and picked[0] == 0 and picked[-1] == 99
    assert overlay.pick(4, 4) == [0, 1, 2, 3]


@pytest.mark.parametrize("p", [None, 0.1, 0.6, 0.95])
def test_overlay_draws_a_band_and_keeps_the_frame_size(p):
    frame = Image.new("RGB", (640, 360), (40, 40, 40))
    out = overlay.draw_verdict(frame, "Put the ring on the peg", p, 0.8)
    assert out.size == frame.size
    assert out.getpixel((10, 10)) == (40, 40, 40)
    assert out.getpixel((10, 355)) != (40, 40, 40)


def plugin_client():
    pytest.importorskip("numpy")
    path = ROOT / "lerobot_plugin" / "lerobot_reward_robojev" / "client.py"
    spec = importlib.util.spec_from_file_location("lerobot_reward_robojev_client", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_plugin_calibration_matches_decision_json():
    plugin = plugin_client()
    assert (plugin.FALSE_TOKEN, plugin.TRUE_TOKEN) == (DECISION["tokens"]["false"], DECISION["tokens"]["true"])
    assert plugin.BIAS == {plugin.FALSE_TOKEN: DECISION["bias"]["false"], plugin.TRUE_TOKEN: DECISION["bias"]["true"]}
    assert plugin.TEMPERATURE == DECISION["temperature"]
    assert plugin.SHORT_SIDE == DECISION["image_short_side"]
    assert plugin.FRAMES == DECISION["frames"]


def test_plugin_asks_the_same_question_as_the_client():
    plugin = plugin_client()
    assert plugin.DENSE_QUESTION == client.DENSE_QUESTION
    assert plugin.TEMPLATE == client.TEMPLATE
