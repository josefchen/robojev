import base64
import io
import json
import math
import shutil
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest
from PIL import Image

import robojev
from robojev import RoboJev, video_frames

DECISION = json.loads((Path(robojev.__file__).parent / "decision.json").read_text())
FALSE, TRUE = DECISION["tokens"]["false"], DECISION["tokens"]["true"]
TRUE_LOGPROB, FALSE_LOGPROB = -0.1, -2.4


@pytest.fixture
def server():
    seen = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append({"path": self.path, "headers": dict(self.headers), "body": body})
            top = [{"token": f"token_id:{TRUE}", "logprob": TRUE_LOGPROB},
                   {"token": f"token_id:{FALSE}", "logprob": FALSE_LOGPROB}]
            data = json.dumps({"choices": [{"logprobs": {"content": [{"top_logprobs": top}]}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    httpd = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_port}", seen
    httpd.shutdown()


def frames(count, size=(320, 240)):
    return [Image.new("RGB", size, (i * 25 % 256, 80, 160)) for i in range(count)]


def calibrated(true_logprob, false_logprob):
    no = (false_logprob + DECISION["bias"]["false"]) / DECISION["temperature"]
    yes = (true_logprob + DECISION["bias"]["true"]) / DECISION["temperature"]
    return 1 / (1 + math.exp(no - yes))


def images(request):
    return [c for c in request["body"]["messages"][0]["content"] if c["type"] == "image_url"]


def test_release_reads_ten_frames():
    assert DECISION["frames"] == 10
    assert RoboJev().frames == 10


def test_done_applies_the_calibration_to_the_two_answer_tokens(server):
    url, seen = server
    p = RoboJev(url).done("Put the ring on the peg", frames(10))
    assert p == pytest.approx(calibrated(TRUE_LOGPROB, FALSE_LOGPROB))
    body = seen[0]["body"]
    assert seen[0]["path"] == "/v1/chat/completions"
    assert body["model"] == "robojev" and body["max_tokens"] == 1
    assert body["allowed_token_ids"] == [FALSE, TRUE]
    content = body["messages"][0]["content"]
    assert "Task given to the robot arm: Put the ring on the peg" in content[0]["text"]
    assert "Images 1-10: main camera" in content[0]["text"]
    assert content[-1]["text"].endswith("[decision]:")
    assert len(images(seen[0])) == 10


def test_frames_are_resized_to_the_calibrated_short_side(server):
    url, seen = server
    RoboJev(url).done("Put the ring on the peg", frames(10, size=(640, 360)))
    data = images(seen[0])[0]["image_url"]["url"].split(",", 1)[1]
    with Image.open(io.BytesIO(base64.b64decode(data))) as image:
        assert min(image.size) == DECISION["image_short_side"]


def test_other_cameras_add_their_last_frame(server):
    url, seen = server
    RoboJev(url).done("Put the ring on the peg", frames(10), views={"wrist": frames(1)[0]})
    assert len(images(seen[0])) == 11
    assert "Image 11: wrist camera at the end." in seen[0]["body"]["messages"][0]["content"][0]["text"]


def test_headers_reach_the_server(server):
    url, seen = server
    RoboJev(url, headers={"Authorization": "Bearer test"}).done("Put the ring on the peg", frames(10))
    assert seen[0]["headers"]["Authorization"] == "Bearer test"


def test_wrong_frame_count_is_rejected():
    with pytest.raises(ValueError, match="reads 10 frames"):
        RoboJev("http://127.0.0.1:9").done("Put the ring on the peg", frames(4))


needs_ffmpeg = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="needs ffmpeg")


@pytest.fixture
def attempt(tmp_path):
    video = tmp_path / "attempt.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=duration=3:size=320x240:rate=10",
                    "-pix_fmt", "yuv420p", str(video)], check=True)
    return video


@needs_ffmpeg
def test_video_frames_cuts_evenly_spaced_frames(attempt):
    cut = video_frames(attempt, 10)
    assert len(cut) == 10
    assert all(frame.size == (320, 240) for frame in cut)


@needs_ffmpeg
def test_done_cuts_a_video_into_the_frames_the_release_reads(server, attempt):
    url, seen = server
    RoboJev(url).done("Put the ring on the peg", attempt)
    assert len(images(seen[0])) == 10
