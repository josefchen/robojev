"""Ask a RoboJev server whether a task was completed, from frames already in memory."""

from __future__ import annotations

import base64
import io
import math
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import requests
from PIL import Image

# The released model's decision.json; repeated here so the client needs no download.
FALSE_TOKEN, TRUE_TOKEN = 3721, 1802
BIAS = {FALSE_TOKEN: 0.00011960109259234741, TRUE_TOKEN: -0.00011981795250903815}
TEMPERATURE = 0.9
SHORT_SIDE = 288
FRAMES = 10  # main-camera frames the released model reads, evenly spaced from start to end

QUESTION = ("The images are frames from a robot arm's attempt at the task in the state, in time order; "
            "the last image shows the end of the attempt. Did the robot complete the task? "
            "True means: By the end of the attempt the task is completed as instructed. "
            "False means: The task is not completed: wrong place, dropped, not attempted, done wrongly or still in progress.")
STATE = ("Task given to the robot arm: {task}\nFrames from one camera in time order (image 1: start of the attempt, "
         "image 2: a third of the way, image 3: two thirds of the way, image 4: end of the attempt).")
DENSE_QUESTION = (
    "The images are frames from a robot arm's attempt at the task in the state. The first images come from "
    "one camera, evenly spaced in time from the start of the attempt to its end; any further images show the "
    "end of the attempt from other cameras. Did the robot complete the task? "
    "True means: By the end of the attempt the task is completed as instructed. "
    "False means: The task is not completed: wrong place, dropped, not attempted, done wrongly or still in progress."
)
DENSE_STATE = ("Task given to the robot arm: {task}\nImages 1-{frames}: main camera, evenly spaced from the start to "
               "the end of the attempt.")
TEMPLATE = ("{%- for m in messages -%}{%- for c in m['content'] -%}{%- if c['type'] == 'text' -%}{{ c['text'] }}"
            "{%- else -%}<|vision_start|><|image_pad|><|vision_end|>{%- endif -%}{%- endfor -%}{%- endfor -%}")


def to_image(frame: Any) -> Image.Image:
    """A PIL image from a PIL image, JPEG bytes, an HWC or CHW array or tensor (uint8, or float in [0, 1])."""
    if isinstance(frame, Image.Image):
        return frame.convert("RGB")
    if isinstance(frame, bytes):
        return Image.open(io.BytesIO(frame)).convert("RGB")
    array = frame.detach().cpu().numpy() if hasattr(frame, "detach") else np.asarray(frame)
    if array.ndim == 3 and array.shape[0] in (1, 3) and array.shape[-1] not in (1, 3):
        array = np.transpose(array, (1, 2, 0))
    if array.dtype != np.uint8:
        array = np.clip(array * (255.0 if array.max() <= 1.0 else 1.0), 0, 255).astype(np.uint8)
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    return Image.fromarray(array)


def compact(frame: Any) -> bytes:
    """A frame at the size RoboJev reads, as JPEG bytes, for keeping a whole episode in memory."""
    image = to_image(frame)
    scale = SHORT_SIDE / min(image.size)
    image = image.resize((round(image.width * scale), round(image.height * scale)), Image.BICUBIC)
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=95)
    return buffer.getvalue()


def pick_frames(count: int, frames: int = FRAMES) -> list[int]:
    """`frames` indices evenly spaced over a sequence of `count` frames, first and last included."""
    if count < 1:
        raise ValueError("At least one frame is needed")
    return [round(i * (count - 1) / (frames - 1)) for i in range(frames)]


def _jpeg(image: Image.Image) -> str:
    scale = SHORT_SIDE / min(image.size)
    image = image.resize((round(image.width * scale), round(image.height * scale)), Image.BICUBIC)
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=92)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


class RoboJevClient:
    def __init__(self, url: str = "http://localhost:8000", model: str = "robojev", timeout: float = 120.0,
                 headers: dict[str, str] | None = None):
        self.url, self.model, self.timeout = url.rstrip("/"), model, timeout
        self.session = requests.Session()
        self.session.headers.update(headers or {})

    def done(self, task: str, frames: list[Any]) -> float:
        """P(task completed) from FRAMES frames evenly spaced from the start to the end of the attempt."""
        if len(frames) != FRAMES:
            raise ValueError(f"RoboJev reads {FRAMES} frames evenly spaced from the start to the end of the attempt")
        state = STATE.format(task=task) if FRAMES == 4 else DENSE_STATE.format(task=task, frames=FRAMES)
        question = QUESTION if FRAMES == 4 else DENSE_QUESTION
        content = [{"type": "text", "text": f"[kind] noul\n[state] {state}\n"}]
        content += [{"type": "image_url", "image_url": {"url": _jpeg(to_image(f))}} for f in frames]
        content.append({"type": "text", "text": f"\n[question] {question}\n[options]\nfalse\ntrue\n[decision]:"})
        reply = self.session.post(f"{self.url}/v1/chat/completions", timeout=self.timeout, json={
            "model": self.model, "messages": [{"role": "user", "content": content}], "chat_template": TEMPLATE,
            "add_generation_prompt": False, "add_special_tokens": False, "max_tokens": 1, "temperature": 1.0,
            "top_k": 0, "top_p": 1.0, "logprobs": True, "top_logprobs": 2, "allowed_token_ids": [FALSE_TOKEN, TRUE_TOKEN],
            "return_tokens_as_token_ids": True})
        reply.raise_for_status()
        top = reply.json()["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
        logprob = {int(t["token"].split(":")[1]): t["logprob"] for t in top}
        no, yes = ((logprob.get(t, -1e9) + BIAS[t]) / TEMPERATURE for t in (FALSE_TOKEN, TRUE_TOKEN))
        return 1 / (1 + math.exp(no - yes))

    def done_many(self, tasks: list[str], clips: list[list[Any]], workers: int = 8) -> list[float]:
        """The same question for several attempts at once; each clip is FRAMES frames."""
        with ThreadPoolExecutor(workers) as pool:
            return list(pool.map(self.done, tasks, clips))
