"""Draw RoboJev's completion probability on a robot video.

    python robojev_overlay.py "Put the ring on the peg" attempt.mp4 out.mp4
    python robojev_overlay.py "Put the ring on the peg" --camera 0          # live, requires opencv-python

Every 0.5 s of video, RoboJev is queried with frames recorded up to that point, evenly spaced from
the first frame to the latest (as many as decision.json says the model reads). The task, the probability
and a bar are drawn along the bottom of the frame; the label reads "Done" at or above the
threshold and "Not yet" below it.

Requires requests, pillow and ffmpeg, with robojev.py and decision.json from the model repo in
the same folder.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from robojev import RoboJev

INK = (245, 245, 240)
MUTED = (178, 178, 170)
BAND = (18, 18, 20)
TRACK = (62, 62, 66)
DONE = (92, 201, 128)
UNSURE = (240, 186, 74)
NOT_DONE = (214, 214, 218)


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    for name in (["DejaVuSans-Bold.ttf", "Arial Bold.ttf"] if bold else ["DejaVuSans.ttf", "Arial.ttf"]):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def pick(count: int, frames: int = 4) -> list[int]:
    """`frames` indices evenly spaced over `count` frames, first and last included."""
    return [round(i * (count - 1) / (frames - 1)) for i in range(frames)]


def draw_verdict(frame: Image.Image, task: str, p: float | None, threshold: float) -> Image.Image:
    """The frame with a band along the bottom: the task, a probability bar and the verdict."""
    w, h = frame.size
    unit = max(12, h // 24)
    band_h = round(unit * 4.6)
    out = frame.convert("RGB").copy()
    overlay = Image.new("RGBA", out.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)
    d.rectangle([0, h - band_h, w, h], fill=(*BAND, 215))
    out = Image.alpha_composite(out.convert("RGBA"), overlay).convert("RGB")
    d = ImageDraw.Draw(out)
    pad = unit
    small, big = font(round(unit * 1.15)), font(round(unit * 1.9), bold=True)
    line = task
    while d.textlength(line, font=small) > w - 2 * pad and " " in line:
        line = line.rsplit(" ", 1)[0].rstrip(",;:") + "…"
    d.text((pad, h - band_h + pad * 0.55), line, font=small, fill=MUTED)
    if p is None:
        d.text((pad, h - band_h + pad * 2.0), "Waiting", font=big, fill=MUTED)
        return out
    colour = DONE if p >= threshold else UNSURE if p >= 0.5 else NOT_DONE
    label = f"{'Done' if p >= threshold else 'Not yet'}  {p:.2f}"
    d.text((pad, h - band_h + pad * 2.0), label, font=big, fill=colour)
    label_w = d.textlength(label, font=big)
    x0, x1 = pad + label_w + pad * 1.5, w - pad
    y0 = h - band_h + pad * 2.75
    y1 = y0 + unit * 0.8
    if x1 - x0 > unit * 4:
        d.rounded_rectangle([x0, y0, x1, y1], radius=unit // 3, fill=TRACK)
        d.rounded_rectangle([x0, y0, x0 + max(unit * 0.7, (x1 - x0) * p), y1], radius=unit // 3, fill=colour)
        tx = x0 + (x1 - x0) * threshold
        d.line([tx, y0 - unit * 0.25, tx, y1 + unit * 0.25], fill=INK, width=max(1, unit // 8))
    return out


def read_video(path: str, width: int, fps: float) -> list[Image.Image]:
    probe = json.loads(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                       "stream=width,height", "-of", "json", path],
                                      check=True, capture_output=True, text=True).stdout)["streams"][0]
    height = round(probe["height"] * width / probe["width"] / 2) * 2
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-vf", f"fps={fps},scale={width}:{height}",
                          "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], check=True, capture_output=True).stdout
    size = width * height * 3
    return [Image.frombytes("RGB", (width, height), raw[i:i + size]) for i in range(0, len(raw), size)]


def write_video(frames: list[Image.Image], path: str, fps: float) -> None:
    w, h = frames[0].size
    encoder = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
                                "-r", str(fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
                                "-movflags", "+faststart", path], stdin=subprocess.PIPE)
    for f in frames:
        encoder.stdin.write(f.tobytes())
    encoder.stdin.close()
    if encoder.wait() != 0:
        raise RuntimeError("ffmpeg could not write the video")


def from_file(judge: RoboJev, task: str, video: str, out: str, every: float, threshold: float,
              width: int, fps: float) -> list[tuple[float, float]]:
    frames = read_video(video, width, fps)
    step = max(1, round(every * fps))
    asks = list(range(step, len(frames), step)) + [len(frames) - 1]
    with ThreadPoolExecutor(8) as pool:
        answers = dict(zip(asks, pool.map(lambda n: judge.done(task, [frames[i] for i in pick(n + 1, judge.frames)]), asks)))
    shown, latest = [], None
    for n, f in enumerate(frames):
        latest = answers.get(n, latest)
        shown.append(draw_verdict(f, task, latest, threshold))
    write_video(shown, out, fps)
    return [(n / fps, answers[n]) for n in sorted(answers)]


def from_camera(judge: RoboJev, task: str, camera: int, every: float, threshold: float, record: str | None) -> None:
    import cv2  # only the live mode needs OpenCV
    import numpy as np

    capture = cv2.VideoCapture(camera)
    history, shown, state = [], [], {"p": None, "busy": False}

    def ask(clip: list[Image.Image]) -> None:
        state["p"] = judge.done(task, clip)
        state["busy"] = False

    last = time.monotonic()
    print("Press q to stop, r to start a new attempt.")
    while True:
        ok, bgr = capture.read()
        if not ok:
            break
        frame = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        history.append(frame.resize((frame.width * 288 // frame.height, 288)))
        if time.monotonic() - last >= every and not state["busy"] and len(history) >= 4:
            state["busy"], last = True, time.monotonic()
            threading.Thread(target=ask, args=([history[i] for i in pick(len(history), judge.frames)],), daemon=True).start()
        image = draw_verdict(frame, task, state["p"], threshold)
        if record:
            shown.append(image)
        cv2.imshow("RoboJev", cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR))
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("r"):
            history, state["p"] = [], None
    capture.release()
    cv2.destroyAllWindows()
    if record and shown:
        write_video(shown, record, capture.get(cv2.CAP_PROP_FPS) or 30)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("task", help="the instruction the robot was given")
    parser.add_argument("video", nargs="?", help="a recorded attempt")
    parser.add_argument("out", nargs="?", default="robojev_verdict.mp4")
    parser.add_argument("--camera", type=int, help="judge a live camera instead of a file")
    parser.add_argument("--record", help="with --camera: also save the overlaid video here")
    parser.add_argument("--url", default="http://localhost:8000", help="the RoboJev server")
    parser.add_argument("--header", action="append", default=[], help='extra HTTP header, "Name: value" (repeatable)')
    parser.add_argument("--every", type=float, default=0.5, help="seconds between answers")
    parser.add_argument("--threshold", type=float, default=0.8, help="probability shown as done")
    parser.add_argument("--width", type=int, default=960, help="output width in pixels")
    parser.add_argument("--fps", type=float, default=15, help="output frames per second")
    args = parser.parse_args()
    headers = dict(h.split(":", 1) for h in args.header)
    judge = RoboJev(args.url, decision=Path(__file__).with_name("decision.json"),
                    headers={k.strip(): v.strip() for k, v in headers.items()})
    if args.camera is not None:
        from_camera(judge, args.task, args.camera, args.every, args.threshold, args.record)
        return
    if not args.video:
        parser.error("give a video file or --camera")
    for t, p in from_file(judge, args.task, args.video, args.out, args.every, args.threshold, args.width, args.fps):
        print(f"{t:6.2f} s  {p:.3f}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
