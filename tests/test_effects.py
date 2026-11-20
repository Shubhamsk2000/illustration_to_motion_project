import cv2
import numpy as np
import pytest

from conftest import synthetic_scene
from motion_studio.effects import Glow, Mist, _tileable_noise, find_fire, find_light_source
from motion_studio.episode.build import scene_spec
from motion_studio.episode.looks import suggest_look
from motion_studio.layers import _fill_behind
from motion_studio.paths import sample_image
from motion_studio.scene import SceneRenderer, SceneSpec, spec_from_preset


def test_noise_tiles_seamlessly():
    n = _tileable_noise(90, 160, 0.2, np.random.default_rng(0))
    assert n.min() >= 0 and n.max() <= 1
    step_inside = np.abs(np.diff(n, axis=1)).mean()
    across_seam = np.abs(n[:, 0] - n[:, -1]).mean()
    assert across_seam < 4 * step_inside


@pytest.mark.parametrize("kind", ["fog", "smoke"])
def test_mist_loops_and_stays_soft(kind):
    m = Mist(kind, 320, 180, duration=4, loop=True)
    a0, a1 = m.alpha(0.0), m.alpha(1.0)
    assert np.abs(a0 - a1).mean() < 1e-3
    assert 0.01 < a0.mean() < 0.4 and a0.max() <= 0.5


def test_fog_sits_behind_the_subject(tmp_path):
    path, depth = synthetic_scene(tmp_path, 640, 360)
    plain = SceneRenderer(path, SceneSpec(move="static", breathing=0, light=0), size=(320, 180), fps=12,
                          duration=1, depth=depth)
    foggy = SceneRenderer(path, SceneSpec(move="static", breathing=0, light=0, mist="fog"), size=(320, 180),
                          fps=12, duration=1, depth=depth)
    a, b = plain.frame_at(0.5).astype(int), foggy.frame_at(0.5).astype(int)
    subject = cv2.resize(foggy.layers.mid_alpha, (320, 180)) > 0.95
    assert np.abs(a - b)[~subject].mean() > 2 * np.abs(a - b)[subject].mean()


def _flame(img, x, y):
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    for r, col in [(40, (0, 110, 255)), (22, (40, 190, 255)), (10, (180, 240, 255))]:
        m = np.clip(1 - np.sqrt(((xx - x) / (r * 0.7)) ** 2 + ((yy - y) / r) ** 2), 0, 1)[:, :, None]
        img = (img * (1 - m) + np.array(col) * m).astype(np.uint8)
    return img


def test_fire_is_found_over_skin_coloured_areas():
    img = cv2.imread(str(sample_image("king")))  # cartoon face: bright orange, but not a flame
    img = _flame(img, 120, 820)
    x, y = find_fire(img)
    assert abs(x - 120) < 15 and abs(y - 820) < 15


def test_no_fire_or_light_in_a_dull_image():
    img = np.full((200, 300, 3), (60, 70, 80), np.uint8)
    assert find_fire(img) is None and find_light_source(img) is None
    g = Glow("rays", img, duration=4)
    assert g.map is None
    assert g.apply(img, 0.5, lambda m: m) is img


def test_rays_do_not_clip_a_bright_sky():
    img = np.zeros((300, 400, 3), np.uint8)
    img[:120] = 250  # bright sky
    img[120:] = 60   # dark ground
    g = Glow("rays", img, duration=4)
    assert g.map is not None
    lit = g.apply(img, 0.25, lambda m: m)
    assert (lit[:120].astype(int) - img[:120]).mean() < 3      # sky barely changes
    assert (lit[130:200].astype(int) - img[130:200]).mean() > 3  # shafts show over the ground


def test_firelight_flickers():
    img = _flame(np.full((360, 640, 3), 40, np.uint8), 320, 250)
    g = Glow("firelight", img, duration=4)
    levels = [g.intensity(t) for t in np.linspace(0, 1, 40)]
    assert max(levels) - min(levels) > 0.3


@pytest.mark.parametrize("text,note,look", [
    ("At dusk they gather around the fire.", "", "campfire"),
    ("Rain hammers the tents all night.", "", "storm"),
    ("He writes a letter home.", "soldier writing by candlelight", "campfire"),
    ("The legion marches through the snow.", "", "winter"),
    ("Dawn breaks over the camp.", "", "misty"),
    ("The scribe counts the grain.", "", None),
])
def test_suggest_look(text, note, look):
    assert suggest_look(text, note)[0] == look


def test_note_wins_over_narration():
    # The narration mentions fire, but the picture is described as a sunny road.
    assert suggest_look("They had set fire to the bridge.", "sunny road through fields")[0] == "sunlit"


def test_auto_look_keeps_varied_moves_and_yields_to_overrides():
    a = scene_spec(1, "documentary", {}, auto_preset="campfire")
    assert a.glow == "firelight" and a.atmosphere == "embers" and a.move == "pan_right"
    b = scene_spec(1, "documentary", {"preset": "winter"}, auto_preset="campfire")
    assert b.atmosphere == "snow"


@pytest.mark.parametrize("preset", ["misty", "sunlit", "battle", "campfire"])
def test_new_presets_render(tmp_path, preset):
    path, depth = synthetic_scene(tmp_path, 640, 360)
    r = SceneRenderer(path, spec_from_preset(preset), size=(320, 180), fps=12, duration=1, depth=depth)
    f = r.frame_at(0.5)
    assert f.shape == (180, 320, 3)


def test_lama_fill_changes_only_the_hole(with_lama):
    img = cv2.imread(str(sample_image("fox")))
    mask = np.zeros(img.shape[:2], np.uint8)
    mask[400:900, 300:800] = 255
    out = _fill_behind(img, mask)
    far = np.ones(img.shape[:2], bool)
    far[350:950, 250:850] = False
    assert np.array_equal(out[far], img[far])
    assert out[500:800, 400:700].std() > 8  # textured, not a flat smear
