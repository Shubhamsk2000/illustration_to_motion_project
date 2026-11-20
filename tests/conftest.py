import cv2
import numpy as np
import pytest


@pytest.fixture(autouse=True)
def isolated_caches(tmp_path, monkeypatch):
    """Keep tests from writing into output/."""
    import motion_studio.depth as depth
    import motion_studio.scene as scene

    monkeypatch.setattr(scene, "LAYER_CACHE_DIR", tmp_path / "layers")
    monkeypatch.setattr(depth, "DEPTH_CACHE_DIR", tmp_path / "depth")
    # LaMa takes ~5 s per image; tests use the fast classic fill unless they ask for the real one.
    monkeypatch.setenv("MOTION_STUDIO_NO_LAMA", "1")


@pytest.fixture
def with_lama(monkeypatch):
    from motion_studio.models import lama_inpaint

    monkeypatch.delenv("MOTION_STUDIO_NO_LAMA", raising=False)
    probe = lama_inpaint(np.zeros((8, 8, 3), np.uint8), np.zeros((8, 8), np.uint8))
    if probe is None:
        pytest.skip("LaMa model not available (offline?)")


def synthetic_scene(tmp_path, w, h, name="scene.png"):
    """A textured image with a near 'subject' blob, plus a matching depth map (no model needed)."""
    rng = np.random.default_rng(0)
    img = rng.integers(60, 255, (h, w, 3), dtype=np.uint8)
    img = cv2.GaussianBlur(img, (0, 0), 2)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    depth = 0.2 + 0.3 * yy / h
    blob = ((xx - w * 0.5) / (w * 0.15)) ** 2 + ((yy - h * 0.6) / (h * 0.3)) ** 2 < 1
    depth[blob] = 0.9
    path = tmp_path / name
    cv2.imwrite(str(path), img)
    return path, depth.astype(np.float32)
