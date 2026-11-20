import os
import subprocess

import cv2
import numpy as np
import pytest

from conftest import synthetic_scene
from motion_studio.camera import ALL_MOVES
from motion_studio.paths import ROOT, sample_image
from motion_studio.scene import SceneRenderer, SceneSpec, choose_fit, spec_from_preset

GOLDEN = ROOT / "tests" / "golden" / "warriors_documentary_t050.png"


def test_choose_fit():
    assert choose_fit("auto", 16 / 9, 16 / 9) == "crop"
    assert choose_fit("auto", 3 / 2, 16 / 9) == "crop"
    assert choose_fit("auto", 2 / 3, 16 / 9) == "blur"
    assert choose_fit("blur", 16 / 9, 16 / 9) == "blur"


def test_preset_overrides():
    spec = spec_from_preset("campfire", move="pan_left", strength=None)
    assert spec.atmosphere == "embers" and spec.move == "pan_left" and spec.strength == 0.7
    with pytest.raises(ValueError):
        spec_from_preset("nope")


@pytest.mark.parametrize("move", ALL_MOVES)
@pytest.mark.parametrize("dims", [(800, 450), (1000, 450), (800, 600)])
def test_background_always_covers_frame(tmp_path, move, dims):
    """The background plane must cover the whole frame at every moment (no edge ever shows)."""
    path, depth = synthetic_scene(tmp_path, *dims)
    spec = SceneSpec(move=move, strength=2.0, depth=1.5, fit="crop")
    r = SceneRenderer(path, spec, size=(640, 360), fps=12, duration=2, depth=depth)
    ph, pw = r.layers.background.shape[:2]
    corners = np.array([[0, 0, 1], [r.ow, 0, 1], [0, r.oh, 1], [r.ow, r.oh, 1]], dtype=np.float64).T
    from motion_studio.camera import camera_at

    for t in np.linspace(0, 1, 25):
        m = r._matrix(r.p_bg, *camera_at(move, t, spec.strength))
        inv = cv2.invertAffineTransform(m)
        xs, ys = inv @ corners
        assert xs.min() >= -0.5 and xs.max() <= pw + 0.5, (move, t)
        assert ys.min() >= -0.5 and ys.max() <= ph + 0.5, (move, t)


def test_frame_size_and_portrait_backdrop(tmp_path):
    path, depth = synthetic_scene(tmp_path, 400, 600)
    r = SceneRenderer(path, SceneSpec(), size=(640, 360), fps=12, duration=1, depth=depth)
    assert r.fit == "blur"
    frame = r.frame_at(0.5)
    assert frame.shape == (360, 640, 3) and frame.dtype == np.uint8
    # The side backdrop is darker than the picture in the middle.
    assert frame[:, :40].mean() < frame[:, 300:340].mean()


def test_loop_is_seamless(tmp_path):
    path, depth = synthetic_scene(tmp_path, 640, 360)
    spec = SceneSpec(move="drift", atmosphere="snow", loop=True)
    r = SceneRenderer(path, spec, size=(320, 180), fps=12, duration=2, depth=depth)
    diff = np.abs(r.frame_at(0.0).astype(int) - r.frame_at(1.0).astype(int)).mean()
    assert diff < 0.5
    assert r.times()[-1] < 1.0  # the start frame is not repeated at the end


def test_one_way_ends_on_final_pose(tmp_path):
    path, depth = synthetic_scene(tmp_path, 640, 360)
    r = SceneRenderer(path, SceneSpec(move="push_in"), size=(320, 180), fps=12, duration=2, depth=depth)
    assert r.times()[0] == 0.0 and r.times()[-1] == 1.0 and len(r.times()) == 24


@pytest.mark.parametrize("atmosphere", ["dust", "embers", "fireflies", "snow", "rain"])
def test_atmospheres_render(tmp_path, atmosphere):
    path, depth = synthetic_scene(tmp_path, 640, 360)
    r = SceneRenderer(path, SceneSpec(atmosphere=atmosphere), size=(320, 180), fps=12, duration=1, depth=depth)
    plain = SceneRenderer(path, SceneSpec(), size=(320, 180), fps=12, duration=1, depth=depth)
    assert not np.array_equal(r.frame_at(0.3), plain.frame_at(0.3))


def test_render_writes_video(tmp_path):
    path, depth = synthetic_scene(tmp_path, 640, 360)
    out = tmp_path / "clip.mp4"
    stats = SceneRenderer(path, SceneSpec(), size=(320, 180), fps=12, duration=1, depth=depth).render(out)
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-show_entries", "stream=width,height,nb_read_frames",
         "-of", "csv=p=0", str(out)], capture_output=True, text=True, check=True).stdout.strip()
    assert probe == "320,180,12"
    assert stats["frames"] == 12


def test_golden_frame(with_lama):
    """Guards the look of a real sample (real pipeline, with LaMa). Set UPDATE_GOLDEN=1 to accept a change."""
    r = SceneRenderer(sample_image("warriors"), spec_from_preset("documentary"), size=(480, 270), fps=12, duration=2)
    frame = r.frame_at(0.5)
    if os.environ.get("UPDATE_GOLDEN") or not GOLDEN.exists():
        GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(GOLDEN), frame)
        pytest.skip("golden frame written")
    golden = cv2.imread(str(GOLDEN))
    assert np.abs(frame.astype(int) - golden.astype(int)).mean() < 1.5
