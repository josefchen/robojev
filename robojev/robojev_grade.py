"""Grade the episodes of a LeRobot dataset on the Hub.

    python robojev_grade.py owner/eval_dataset --out results.csv

Requires requests, pillow and ffmpeg, with robojev.py and decision.json in the same folder.
LeRobot v3.0 metadata also needs pyarrow.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from contextlib import nullcontext
from email.utils import parsedate_to_datetime
from itertools import count
from pathlib import Path
from urllib.parse import quote, urljoin, urlsplit

import requests
from PIL import Image

from robojev import RoboJev

PREFERRED = ("front", "top", "table", "above", "overhead", "scene", "side", "laptop", "phone", "cam_high",
             "camera1", "image", "main")
NOT_SCENE = ("wrist", "hand", "gripper", "left_arm", "right_arm")
REDIRECTS = (301, 302, 303, 307, 308)


def _url(repo: str, path: str) -> str:
    return f"https://huggingface.co/datasets/{quote(repo, safe='/')}/resolve/main/{quote(path, safe='/')}"


def _retry_after(value: str) -> float:
    """Seconds to wait, accepting both forms of the Retry-After header."""
    try:
        delay = float(value)
    except ValueError:
        delay = parsedate_to_datetime(value).timestamp() - time.time()
    return max(0.0, delay) if math.isfinite(delay) else 1.0


def _request(client: requests.Session, method: str, url: str, missing_ok: bool = False,
             headers: dict[str, str] | None = None, follow_redirects: bool = True) -> requests.Response:
    """Follow Hub redirects without sending HF_TOKEN to storage hosts."""
    for _ in range(10):
        request_headers = dict(headers or {})
        if urlsplit(url).scheme == "https" and urlsplit(url).hostname == "huggingface.co":
            token = os.environ.get("HF_TOKEN")
            if token:
                request_headers["Authorization"] = f"Bearer {token}"
        for attempt in range(5):
            reply = client.request(method, url, headers=request_headers, allow_redirects=False, stream=True, timeout=60)
            if reply.status_code != 429 or attempt == 4:
                break
            try:
                delay = _retry_after(reply.headers.get("Retry-After") or "1")
            except (ValueError, OverflowError):
                delay = 1.0
            reply.close()
            time.sleep(delay)
        if follow_redirects and reply.status_code in REDIRECTS:
            location = reply.headers.get("Location")
            reply.close()
            if not location:
                raise ValueError("Hub redirect has no Location header")
            url = urljoin(url, location)
            continue
        if missing_ok and reply.status_code == 404:
            return reply
        try:
            reply.raise_for_status()
        except requests.RequestException:
            reply.close()
            raise
        return reply
    raise ValueError("Too many Hub redirects")


def _get(client: requests.Session, repo: str, path: str, missing_ok: bool = False) -> bytes | None:
    with _request(client, "GET", _url(repo, path), missing_ok) as reply:
        return None if reply.status_code == 404 else reply.content


def _cdn(client: requests.Session, repo: str, path: str) -> str:
    """Resolve the storage URL and require byte ranges without reading the video body."""
    url = _url(repo, path)
    for _ in range(10):
        with _request(client, "HEAD", url, follow_redirects=False) as reply:
            if reply.status_code not in REDIRECTS:
                break
            location = reply.headers.get("Location")
            if not location:
                raise ValueError("Hub redirect has no Location header")
            url = urljoin(url, location)
        if urlsplit(url).hostname != "huggingface.co":
            break
    else:
        raise ValueError("Too many Hub redirects")
    with _request(client, "GET", url, headers={"Range": "bytes=0-0"}) as reply:
        if reply.status_code != 206 or not reply.headers.get("Content-Range", "").startswith("bytes 0-0/"):
            raise ValueError("Video storage does not support HTTP byte ranges")
        return reply.url


def pick_camera(videos: list[str], camera: str | None = None) -> str:
    """Choose a third-person camera unless a specific video key was requested."""
    if camera is not None:
        if camera not in videos:
            raise ValueError(f"Unknown camera {camera!r}; available: {', '.join(videos) or 'none'}")
        return camera
    scene = [v for v in videos if not any(word in v.lower() for word in NOT_SCENE)]
    for name in PREFERRED:
        for video in scene:
            if name in video.lower().split(".")[-1]:
                return video
    if scene:
        return scene[0]
    raise ValueError(f"No third-person video camera; choose --camera from: {', '.join(videos) or 'none'}")


def frame_times(start: float, end: float, count: int = 4) -> list[float]:
    """`count` evenly spaced times within the episode, ending 0.3 s before its end."""
    if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start:
        raise ValueError(f"Invalid episode video interval: {start}, {end}")
    span = max(0.0, end - start - 0.3)
    return [start + i / (count - 1) * span for i in range(count)]


def _v3_rows(client: requests.Session, repo: str, camera: str) -> list[dict]:
    """Read the consecutive metadata shards, including all chunks."""
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError("LeRobot v3.0 metadata requires pyarrow; install it with: pip install pyarrow") from error
    columns = ["episode_index", "tasks"] + [f"videos/{camera}/{key}" for key in
                                            ("chunk_index", "file_index", "from_timestamp", "to_timestamp")]
    rows = []
    for chunk in count():
        found = False
        for file in count():
            path = f"meta/episodes/chunk-{chunk:03d}/file-{file:03d}.parquet"
            data = _get(client, repo, path, missing_ok=True)
            if data is None:
                break
            found = True
            table = pq.read_table(io.BytesIO(data), columns=columns)
            rows.extend(table.to_pylist())
        if not found:
            return rows


def episodes(client: requests.Session, repo: str, info: dict, camera: str) -> list[dict]:
    """Episode index, task, video path and its start and end times in that video."""
    version = info.get("codebase_version")
    if version not in ("v2.0", "v2.1", "v3.0"):
        raise ValueError(f"Unsupported LeRobot version: {version!r}")
    if version == "v3.0":
        rows = _v3_rows(client, repo, camera)
    else:
        data = _get(client, repo, "meta/episodes.jsonl")
        rows = [json.loads(line) for line in data.decode().splitlines() if line.strip()]
        fps = float(info["fps"])
        size = int(info.get("chunks_size") or 1000)
        if not math.isfinite(fps) or fps <= 0 or size <= 0:
            raise ValueError("LeRobot fps and chunks_size must be positive")
    out = []
    for row in rows:
        index = int(row["episode_index"])
        if version == "v3.0":
            prefix = f"videos/{camera}"
            path = info["video_path"].format(video_key=camera, chunk_index=row[f"{prefix}/chunk_index"],
                                           file_index=row[f"{prefix}/file_index"])
            start, end = float(row[f"{prefix}/from_timestamp"]), float(row[f"{prefix}/to_timestamp"])
        else:
            path = info["video_path"].format(episode_chunk=index // size, video_key=camera, episode_index=index)
            start, end = 0.0, float(row["length"]) / fps
        frame_times(start, end)
        tasks = row.get("tasks") or []
        if not isinstance(tasks, list) or not tasks or not isinstance(tasks[0], str) or not tasks[0].strip():
            raise ValueError(f"Episode {index} has no task text")
        out.append({"episode": index, "task": tasks[0], "path": path, "start": start, "end": end})
    if len({row["episode"] for row in out}) != len(out):
        raise ValueError("Duplicate episode indices in metadata")
    if "total_episodes" in info and len(out) != int(info["total_episodes"]):
        raise ValueError("Episode metadata count differs from total_episodes")
    return sorted(out, key=lambda row: row["episode"])


def video_frames(url: str, start: float, end: float, count: int = 4) -> list[Image.Image]:
    """Seek directly into the remote video and decode only the `count` frames the model reads."""
    with tempfile.TemporaryDirectory() as folder:
        command = ["ffmpeg", "-v", "error", "-nostdin", "-y"]
        for point in frame_times(start, end, count):
            command += ["-ss", f"{point:.3f}", "-seekable", "1", "-rw_timeout", "30000000", "-i", url]
        for i in range(count):
            command += ["-map", f"{i}:v:0", "-frames:v", "1", "-q:v", "2", str(Path(folder) / f"{i}.jpg")]
        try:
            subprocess.run(command, check=True, capture_output=True, timeout=180)
        except subprocess.CalledProcessError as error:
            raise RuntimeError("ffmpeg could not read the episode frames") from error
        frames = []
        for i in range(count):
            with Image.open(Path(folder) / f"{i}.jpg") as image:
                frames.append(image.convert("RGB"))
        return frames


def grade(client: requests.Session, repo: str, rows: list[dict], judge: RoboJev | None, threshold: float,
          out: Path | None = None) -> None:
    """Print each result and a summary, optionally saving the results as CSV."""
    done, unsure = 0, 0
    with out.open("w", newline="", encoding="utf-8") if out else nullcontext() as stream:
        writer = csv.writer(stream) if stream else None
        if writer:
            writer.writerow(["episode", "task", "p_done", "verdict"])
        print("Episode\tTask\tP(done)\tVerdict")
        for row in rows:
            url = _cdn(client, repo, row["path"])
            frames = video_frames(url, row["start"], row["end"], getattr(judge, "frames", 4))
            try:
                probability = float(judge.done(row["task"], frames))
            finally:
                for frame in frames:
                    frame.close()
            if not 0 <= probability <= 1:
                raise ValueError(f"Episode {row['episode']}: server probability is outside [0, 1]")
            passed = probability >= threshold
            verdict = "Done" if passed else "Not yet"
            task = " ".join(row["task"].split())
            print(f"{row['episode']}\t{task}\t{probability:.3f}\t{verdict}", flush=True)
            if writer:
                writer.writerow([row["episode"], row["task"], probability, verdict])
                stream.flush()
            done += passed
            unsure += 0.2 <= probability <= 0.8
    count = len(rows)
    print(f"Episodes graded: {count}")
    rate = f"{done / count:.1%}" if count else "n/a"
    print(f"Success rate at threshold {threshold:g}: {rate} ({done}/{count})")
    print(f"Worth a look (0.2 <= P(done) <= 0.8): {unsure}")


def _indices(value: str) -> set[int]:
    try:
        indices = {int(part) for part in value.split(",")}
    except ValueError as error:
        raise argparse.ArgumentTypeError("use comma-separated episode indices, such as 0,1,5") from error
    if any(index < 0 for index in indices):
        raise argparse.ArgumentTypeError("episode indices must be nonnegative")
    return indices


def _threshold(value: str) -> float:
    try:
        threshold = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("threshold must be between 0 and 1") from error
    if not 0 <= threshold <= 1:
        raise argparse.ArgumentTypeError("threshold must be between 0 and 1")
    return threshold


def _header(value: str) -> tuple[str, str]:
    name, separator, content = value.partition(":")
    name = name.strip()
    if not separator or not name or any(c.isspace() for c in name) or "\n" in value or "\r" in value:
        raise argparse.ArgumentTypeError('headers must use "Name: value"')
    return name, content.strip()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("repo", help="LeRobot dataset repository, such as owner/eval_dataset")
    parser.add_argument("--server", default="http://localhost:8000", help="the RoboJev server")
    parser.add_argument("--camera", help="video feature key; defaults to a third-person camera")
    parser.add_argument("--episodes", type=_indices, help="comma-separated episode indices")
    parser.add_argument("--threshold", type=_threshold, default=0.5, help="minimum P(done) for Done (default: 0.5)")
    parser.add_argument("--out", type=Path, help="write episode results to a CSV file")
    parser.add_argument("--header", type=_header, action="append", default=[],
                        help='RoboJev server header, "Name: value" (repeatable)')
    args = parser.parse_args(argv)
    try:
        with requests.Session() as client:
            info = json.loads(_get(client, args.repo, "meta/info.json"))
            videos = [key for key, feature in info["features"].items() if feature.get("dtype") == "video"]
            camera = pick_camera(videos, args.camera)
            rows = episodes(client, args.repo, info, camera)
            if args.episodes is not None:
                missing = args.episodes - {row["episode"] for row in rows}
                if missing:
                    raise ValueError(f"Unknown episode indices: {', '.join(map(str, sorted(missing)))}")
                rows = [row for row in rows if row["episode"] in args.episodes]
            print(f"Camera: {camera}", file=sys.stderr)
            judge = RoboJev(args.server, headers=dict(args.header)) if rows else None
            grade(client, args.repo, rows, judge, args.threshold, args.out)
    except (OSError, ValueError, KeyError, RuntimeError, requests.RequestException, subprocess.SubprocessError) as error:
        parser.exit(1, f"error: {error}\n")


if __name__ == "__main__":
    main()
