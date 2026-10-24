import argparse
import csv
import json

import pytest
from PIL import Image

from robojev import robojev_grade as grade

CAMERA = "observation.images.front"
TASK = "Put the ring on the peg"


def test_frame_times_are_even_and_stop_before_the_end():
    times = grade.frame_times(2.0, 12.0, 10)
    assert len(times) == 10
    assert times[0] == 2.0 and times[-1] == pytest.approx(11.7)
    steps = [b - a for a, b in zip(times, times[1:])]
    assert max(steps) - min(steps) < 1e-9


@pytest.mark.parametrize("start, end", [(5.0, 5.0), (-1.0, 3.0), (0.0, float("nan"))])
def test_frame_times_reject_bad_intervals(start, end):
    with pytest.raises(ValueError):
        grade.frame_times(start, end)


def test_pick_camera_prefers_a_third_person_view():
    assert grade.pick_camera(["observation.images.wrist", CAMERA]) == CAMERA
    assert grade.pick_camera(["observation.images.gripper", "observation.images.cam_high"]) == "observation.images.cam_high"


def test_pick_camera_needs_a_scene_camera_or_an_explicit_choice():
    with pytest.raises(ValueError, match="No third-person video camera"):
        grade.pick_camera(["observation.images.wrist"])
    assert grade.pick_camera(["observation.images.wrist"], "observation.images.wrist") == "observation.images.wrist"
    with pytest.raises(ValueError, match="Unknown camera"):
        grade.pick_camera([CAMERA], "observation.images.top")


def test_v2_episodes_have_one_video_file_each(monkeypatch):
    info = {"codebase_version": "v2.1", "fps": 30, "chunks_size": 1000, "total_episodes": 2,
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"}
    lines = [{"episode_index": 1, "tasks": [TASK], "length": 90}, {"episode_index": 0, "tasks": [TASK], "length": 60}]
    monkeypatch.setattr(grade, "_get", lambda client, repo, path, missing_ok=False:
                        "\n".join(json.dumps(line) for line in lines).encode())
    rows = grade.episodes(None, "owner/eval_dataset", info, CAMERA)
    assert [row["episode"] for row in rows] == [0, 1]
    assert rows[1]["path"] == f"videos/chunk-000/{CAMERA}/episode_000001.mp4"
    assert (rows[1]["start"], rows[1]["end"]) == (0.0, 3.0)


def v3_row(index, file_index, start, end):
    prefix = f"videos/{CAMERA}"
    return {"episode_index": index, "tasks": [TASK], f"{prefix}/chunk_index": 0, f"{prefix}/file_index": file_index,
            f"{prefix}/from_timestamp": start, f"{prefix}/to_timestamp": end}


V3 = {"codebase_version": "v3.0", "total_episodes": 3,
      "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"}


def test_v3_episodes_resolve_their_offsets_in_shared_video_files(monkeypatch):
    rows = [v3_row(0, 0, 0.0, 8.0), v3_row(1, 0, 8.0, 15.5), v3_row(2, 1, 0.0, 6.0)]
    monkeypatch.setattr(grade, "_v3_rows", lambda client, repo, camera: rows)
    out = grade.episodes(None, "owner/eval_dataset", V3, CAMERA)
    assert out[0]["path"] == out[1]["path"] == f"videos/{CAMERA}/chunk-000/file-000.mp4"
    assert (out[1]["start"], out[1]["end"]) == (8.0, 15.5)
    assert out[2]["path"] == f"videos/{CAMERA}/chunk-000/file-001.mp4"


def test_episode_count_must_match_the_metadata(monkeypatch):
    monkeypatch.setattr(grade, "_v3_rows", lambda client, repo, camera: [v3_row(0, 0, 0.0, 8.0)])
    with pytest.raises(ValueError, match="total_episodes"):
        grade.episodes(None, "owner/eval_dataset", V3, CAMERA)


def test_unsupported_versions_are_rejected():
    with pytest.raises(ValueError, match="Unsupported LeRobot version"):
        grade.episodes(None, "owner/eval_dataset", {"codebase_version": "v1.6"}, CAMERA)


def test_grade_prints_a_summary_and_writes_csv(monkeypatch, tmp_path, capsys):
    answers = iter([0.9, 0.5, 0.1])

    class Judge:
        frames = 10

        def done(self, task, frames):
            assert task == TASK and len(frames) == 10
            return next(answers)

    monkeypatch.setattr(grade, "_cdn", lambda client, repo, path: f"https://storage.example/{path}")
    monkeypatch.setattr(grade, "video_frames", lambda url, start, end, count=4: [Image.new("RGB", (8, 8))] * count)
    rows = [{"episode": i, "task": TASK, "path": f"episode_{i}.mp4", "start": 0.0, "end": 5.0} for i in range(3)]
    grade.grade(None, "owner/eval_dataset", rows, Judge(), 0.5, tmp_path / "results.csv")
    printed = capsys.readouterr().out
    assert "Success rate at threshold 0.5: 66.7% (2/3)" in printed
    assert "Worth a look (0.2 <= P(done) <= 0.8): 1" in printed
    with (tmp_path / "results.csv").open() as stream:
        assert [row["verdict"] for row in csv.DictReader(stream)] == ["Done", "Done", "Not yet"]


def test_argument_parsers():
    assert grade._header("Authorization: Bearer token") == ("Authorization", "Bearer token")
    assert grade._indices("0,2,5") == {0, 2, 5}
    assert grade._threshold("0.8") == 0.8
    for parse, value in [(grade._header, "no colon"), (grade._indices, "-1"), (grade._threshold, "1.5")]:
        with pytest.raises(argparse.ArgumentTypeError):
            parse(value)


def test_retry_after_accepts_seconds_and_dates():
    assert grade._retry_after("2") == 2.0
    assert grade._retry_after("Wed, 21 Oct 2015 07:28:00 GMT") == 0.0
