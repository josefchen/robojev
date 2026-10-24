"""Ask RoboJev whether a robot finished its task.

Start the model once:

    vllm serve josefchen/RoboJev-27B --served-model-name robojev --max-model-len 32768 \
        --limit-mm-per-prompt '{"image": 16}' --logprobs-mode processed_logprobs --trust-request-chat-template

Then:

    from robojev import RoboJev
    judge = RoboJev("http://localhost:8000")
    judge.done("Put the ring on the peg", "attempt.mp4")          # frames cut evenly from a video
    judge.done("Put the ring on the peg", "attempt.mp4", views={"wrist": "wrist_end.jpg"})

Both return the probability that the task was completed, between 0 and 1. decision.json says how many
frames the model reads from the main camera (`frames`; four when absent); `views` adds the final
frame from other cameras. Needs: requests, pillow, and ffmpeg for videos.
"""

from __future__ import annotations

import base64
import io
import json
import math
import subprocess
import tempfile
from pathlib import Path

import requests
from PIL import Image

QUESTION = ("The images are frames from a robot arm's attempt at the task in the state, in time order; "
            "the last image shows the end of the attempt. Did the robot complete the task? "
            "True means: By the end of the attempt the task is completed as instructed. "
            "False means: The task is not completed: wrong place, dropped, not attempted, done wrongly or still in progress.")
STATE = ("Task given to the robot arm: {task}\nFrames from one camera in time order (image 1: start of the attempt, "
         "image 2: a third of the way, image 3: two thirds of the way, image 4: end of the attempt).")
CAMERA_NAMES = {"front": "front camera", "top": "overhead camera", "wrist": "wrist camera"}
VIEWS_QUESTION = (
    "The first four images are frames from one camera across a robot arm's attempt at the task in the state, "
    "in time order; the remaining images show the end of the attempt from other cameras. "
    "Did the robot complete the task? "
    "True means: By the end of the attempt the task is completed as instructed. "
    "False means: The task is not completed: wrong place, dropped, not attempted, done wrongly or still in progress."
)
DENSE_QUESTION = (
    "The images are frames from a robot arm's attempt at the task in the state. The first images come from "
    "one camera, evenly spaced in time from the start of the attempt to its end; any further images show the "
    "end of the attempt from other cameras. Did the robot complete the task? "
    "True means: By the end of the attempt the task is completed as instructed. "
    "False means: The task is not completed: wrong place, dropped, not attempted, done wrongly or still in progress."
)
TEMPLATE = ("{%- for m in messages -%}{%- for c in m['content'] -%}{%- if c['type'] == 'text' -%}{{ c['text'] }}"
            "{%- else -%}<|vision_start|><|image_pad|><|vision_end|>{%- endif -%}{%- endfor -%}{%- endfor -%}")


def camera_name(camera: str) -> str:
    """How the prompt names a camera: the known names, else the last part of a LeRobot video key."""
    short = camera.split(".")[-1]
    return CAMERA_NAMES.get(camera) or CAMERA_NAMES.get(short) or f"{short} camera"


def views_state_text(instruction: str, cameras: list[str]) -> str:
    """Four frames from the first camera, then the final frame from each other camera."""
    if len(cameras) == 1:
        return STATE.format(task=instruction)
    names = [camera_name(c) for c in cameras]
    images = [f"Images 1-4: {names[0]} at the start, a third, two thirds and the end of the attempt."]
    images += [f"Image {i + 5}: {name} at the end." for i, name in enumerate(names[1:])]
    return f"Task given to the robot arm: {instruction}\n" + "\n".join(images)


def dense_state_text(instruction: str, cameras: list[str], frames: int = 10) -> str:
    """`frames` frames from the first camera, start to end, then the final frame from each other camera."""
    names = [camera_name(c) for c in cameras]
    images = [f"Images 1-{frames}: {names[0]}, evenly spaced from the start to the end of the attempt."]
    images += [f"Image {frames + i + 1}: {name} at the end." for i, name in enumerate(names[1:])]
    return f"Task given to the robot arm: {instruction}\n" + "\n".join(images)


def video_frames(video: str | Path, count: int = 4) -> list[Image.Image]:
    """Evenly spaced frames from start to end, the last one 0.3 s before the end of the file."""
    duration = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                                     str(video)], check=True, capture_output=True, text=True).stdout)
    frames = []
    with tempfile.TemporaryDirectory() as folder:
        for i in range(count):
            out = Path(folder) / f"{i}.jpg"
            seek = ["-sseof", "-0.3"] if i == count - 1 else ["-ss", f"{i / (count - 1) * max(0.0, duration - 0.3):.3f}"]
            subprocess.run(["ffmpeg", "-v", "error", "-y", *seek, "-i", str(video), "-frames:v", "1", "-q:v", "2",
                            str(out)], check=True)
            frames.append(Image.open(out).convert("RGB"))
    return frames


def _jpeg(image: Image.Image, short_side: int) -> str:
    scale = short_side / min(image.size)
    image = image.resize((round(image.width * scale), round(image.height * scale)), Image.BICUBIC)
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=92)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


class RoboJev:
    def __init__(self, url: str = "http://localhost:8000", model: str = "robojev", decision: str | Path | None = None,
                 headers: dict[str, str] | None = None):
        self.url, self.model, self.headers = url.rstrip("/"), model, headers or {}
        path = Path(decision) if decision else Path(__file__).with_name("decision.json")
        self.decision = json.loads(path.read_text())
        self.frames = int(self.decision.get("frames", 4))  # main-camera frames per question

    def done(self, task: str, attempt: str | Path | list,
             views: dict[str, Image.Image | str | Path] | None = None, camera: str = "main") -> float:
        """Judge a main-camera attempt, optionally with final frames from other cameras.
        `camera` names the main camera in the prompt."""
        frames = video_frames(attempt, self.frames) if isinstance(attempt, (str, Path)) else \
            [f if isinstance(f, Image.Image) else Image.open(f).convert("RGB") for f in attempt]
        if len(frames) != self.frames:
            raise ValueError(f"This RoboJev release reads {self.frames} frames evenly spaced from the start to the end "
                             f"of the attempt; got {len(frames)}. Pass a video file to have them cut for you.")
        if self.frames != 4:
            state, question = dense_state_text(task, [camera, *(views or {})], self.frames), DENSE_QUESTION
        elif views:
            state, question = views_state_text(task, [camera, *views]), VIEWS_QUESTION
        else:
            state, question = STATE.format(task=task), QUESTION
        if views:
            frames += [f if isinstance(f, Image.Image) else Image.open(f).convert("RGB") for f in views.values()]
        d = self.decision
        content = [{"type": "text", "text": f"[kind] noul\n[state] {state}\n"}]
        content += [{"type": "image_url", "image_url": {"url": _jpeg(f, d["image_short_side"])}} for f in frames]
        content.append({"type": "text", "text": f"\n[question] {question}\n[options]\nfalse\ntrue\n[decision]:"})
        ids = [d["tokens"]["false"], d["tokens"]["true"]]
        reply = requests.post(f"{self.url}/v1/chat/completions", timeout=120, headers=self.headers, json={
            "model": self.model, "messages": [{"role": "user", "content": content}], "chat_template": TEMPLATE,
            "add_generation_prompt": False, "add_special_tokens": False, "max_tokens": 1, "temperature": 1.0,
            "top_k": 0, "top_p": 1.0, "logprobs": True, "top_logprobs": 2, "allowed_token_ids": ids,
            "return_tokens_as_token_ids": True})
        reply.raise_for_status()
        top = reply.json()["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
        logprob = {int(t["token"].split(":")[1]): t["logprob"] for t in top}
        no, yes = ((logprob.get(i, -1e9) + d["bias"][k]) / d["temperature"] for i, k in zip(ids, ("false", "true")))
        return 1 / (1 + math.exp(no - yes))


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        sys.exit("usage: python robojev.py 'task' attempt.mp4 | frame1.jpg ... (start to end, as many as decision.json says)")
    attempt = sys.argv[2] if len(sys.argv) == 3 else sys.argv[2:]
    print(f"{RoboJev().done(sys.argv[1], attempt):.3f}")
