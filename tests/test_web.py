import subprocess
import sys
import time

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from motion_studio.web import app as web
from motion_studio.web.jobs import JobManager

SCRIPT = """# Test Episode

## One

The first scene by the campfire.
> Image: soldiers by a fire

The second scene on the road.

## Two

The third scene.
"""


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("MOTION_STUDIO_EPISODES", str(tmp_path / "episodes"))
    return TestClient(web.app)


@pytest.fixture
def ep(client):
    assert client.post("/api/episodes", json={"name": "test-ep", "title": "Test"}).status_code == 200
    r = client.put("/api/episodes/test-ep/script", json={"text": SCRIPT})
    assert r.status_code == 200
    return r.json()


def png_bytes(w=320, h=180):
    img = np.random.default_rng(0).integers(0, 255, (h, w, 3), dtype=np.uint8)
    return cv2.imencode(".png", img)[1].tobytes()


def test_create_list_and_names(client, ep):
    assert [e["name"] for e in client.get("/api/episodes").json()] == ["test-ep"]
    assert client.post("/api/episodes", json={"name": "test-ep"}).status_code == 400   # exists
    assert client.post("/api/episodes", json={"name": "Bad Name!"}).status_code == 400
    assert client.get("/api/episodes/nope").status_code == 404


def test_state_has_scenes_and_auto_looks(ep):
    assert ep["title"] == "Test Episode"
    assert [s["id"] for s in ep["scenes"]] == ["001", "002", "003"]
    assert ep["scenes"][0]["auto"] == {"look": "campfire", "word": "fire"}
    assert ep["scenes"][0]["spec"]["glow"] == "firelight"
    assert ep["summary"]["chapters"] == ["One", "Two"]


def test_scene_look_edit_validate_and_reset(client, ep):
    r = client.put("/api/episodes/test-ep/scenes/002", json={"preset": "winter", "strength": 1.5})
    s = r.json()["scenes"][1]
    assert s["override"] == {"preset": "winter", "strength": 1.5} and s["spec"]["atmosphere"] == "snow"
    assert client.put("/api/episodes/test-ep/scenes/002", json={"move": "spin"}).status_code == 400
    assert client.put("/api/episodes/test-ep/scenes/002", json={"strength": 9}).status_code == 400
    assert client.put("/api/episodes/test-ep/scenes/099", json={}).status_code == 400
    r = client.put("/api/episodes/test-ep/scenes/002", json={})
    assert r.json()["scenes"][1]["override"] == {}


def test_settings(client, ep):
    r = client.put("/api/episodes/test-ep/settings", json={"transition": "0.9", "auto_look": False})
    st = r.json()["settings"]
    assert st["transition"] == 0.9 and st["auto_look"] is False
    assert r.json()["scenes"][0]["auto"] is None
    assert client.put("/api/episodes/test-ep/settings", json={"bogus": 1}).status_code == 400
    assert client.put("/api/episodes/test-ep/settings", json={"default_preset": "x"}).status_code == 400


def test_image_upload_replaces_and_validates(client, ep):
    up = lambda name, data: client.post("/api/episodes/test-ep/scenes/2/image", files={"file": (name, data)})  # noqa: E731
    r = up("first.jpg", cv2.imencode(".jpg", np.zeros((90, 160, 3), np.uint8))[1].tobytes())
    assert r.status_code == 200 and r.json()["scenes"][1]["image"]["name"] == "002.jpg"
    r = up("second.png", png_bytes())
    assert r.json()["scenes"][1]["image"]["name"] == "002.png"
    assert r.json()["summary"]["images_found"] == 1  # the jpg was replaced, not kept alongside
    assert up("notes.png", b"not an image").status_code == 400
    assert up("doc.pdf", png_bytes()).status_code == 400
    assert client.post("/api/episodes/test-ep/scenes/9/image", files={"file": ("a.png", png_bytes())}).status_code == 404
    thumb = r.json()["scenes"][1]["image"]["thumb"]
    assert client.get(thumb).status_code == 200


def test_audio_upload(client, ep, tmp_path):
    wav = tmp_path / "v.wav"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "sine=d=2", str(wav)], check=True)
    r = client.post("/api/episodes/test-ep/audio/narration", files={"file": ("my voice.wav", wav.read_bytes())})
    n = r.json()["narration"]
    assert n["name"] == "narration.wav" and n["seconds"] == pytest.approx(2.0, abs=0.1)
    assert client.get(n["url"]).status_code == 200
    assert client.post("/api/episodes/test-ep/audio/other", files={"file": ("a.wav", b"x")}).status_code == 400


def test_files_are_confined_to_the_episode(client, ep):
    for path in ["script.md", "episode.yaml", "../../etc/passwd", "images/../episode.yaml", ".build/job/build.log"]:
        assert client.get(f"/files/test-ep/{path}").status_code == 404, path


def test_preview_of_a_placeholder_scene(client, ep):
    r = client.post("/api/episodes/test-ep/scenes/003/preview")
    assert r.status_code == 200 and r.json()["placeholder"] is True
    video = client.get(r.json()["url"])
    assert video.status_code == 200 and len(video.content) > 1000


def test_build_needs_narration(client, ep):
    r = client.post("/api/episodes/test-ep/build", json={"draft": True})
    assert r.status_code == 400 and "narration" in r.json()["detail"]


FAKE_BUILD = """
import json, sys, time
path = sys.argv[1]
for i in range(1, 4):
    json.dump({"stage": "render", "progress": i / 4, "detail": f"scene {i} of 3"}, open(path, "w"))
    time.sleep(0.3)
if "fail" in sys.argv:
    print("Error: something went wrong", file=sys.stderr); sys.exit(1)
json.dump({"seconds": 10}, open(path + ".result.json", "w"))
"""


def fake_jobs(extra=()):
    return JobManager(lambda folder, draft, progress: [sys.executable, "-c", FAKE_BUILD, str(progress), *extra])


def wait_done(jm, name):
    for _ in range(100):
        st = jm.status(name)
        if st["state"] != "running":
            return st
        time.sleep(0.1)
    raise AssertionError("job did not finish")


def test_job_progress_result_and_one_at_a_time(tmp_path):
    jm = fake_jobs()
    jm.start("a", tmp_path / "a", draft=True)
    with pytest.raises(RuntimeError):
        jm.start("b", tmp_path / "b", draft=True)
    time.sleep(0.5)
    mid = jm.status("a")
    assert mid["state"] == "running" and 0 < mid["progress"] < 1 and mid["detail"].startswith("scene")
    st = wait_done(jm, "a")
    assert st["state"] == "done" and st["progress"] == 1.0 and st["result"] == {"seconds": 10}


def test_job_failure_and_cancel(tmp_path):
    jm = fake_jobs(["fail"])
    jm.start("a", tmp_path / "a", draft=False)
    st = wait_done(jm, "a")
    assert st["state"] == "failed" and st["error"] == "something went wrong"

    jm = JobManager(lambda folder, draft, progress: [sys.executable, "-c", "import time; time.sleep(30)"])
    jm.start("b", tmp_path / "b", draft=True)
    assert jm.cancel("b") is True
    assert jm.status("b")["state"] == "cancelled"
    assert jm.cancel("b") is False
