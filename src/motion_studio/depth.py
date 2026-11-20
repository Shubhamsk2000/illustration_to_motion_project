"""Depth estimation (Depth-Anything-V2-Small, CPU) with an on-disk cache keyed by image content."""

import hashlib
from pathlib import Path

import cv2
import numpy as np

from motion_studio.paths import DEPTH_CACHE_DIR, DEPTH_DIR, SAMPLES_DIR

_DEPTH_PIPE = None


def get_depth_pipeline():
    global _DEPTH_PIPE
    if _DEPTH_PIPE is None:
        from transformers import pipeline

        print("[depth] Loading Depth-Anything-V2-Small model...")
        _DEPTH_PIPE = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")
    return _DEPTH_PIPE


def estimate_depth(image_bgr_or_pil):
    """Normalized depth map as float32 in [0, 1], where 1.0 is closest to the camera."""
    from PIL import Image

    if isinstance(image_bgr_or_pil, np.ndarray):
        pil_img = Image.fromarray(cv2.cvtColor(image_bgr_or_pil, cv2.COLOR_BGR2RGB))
    else:
        pil_img = image_bgr_or_pil

    depth = np.array(get_depth_pipeline()(pil_img)["depth"], dtype=np.float32)
    d_min, d_max = float(depth.min()), float(depth.max())
    if d_max > d_min:
        return (depth - d_min) / (d_max - d_min)
    return np.zeros_like(depth)


def file_hash(path) -> str:
    return hashlib.sha1(Path(path).read_bytes()).hexdigest()[:16]


def depth_for_image(image_path, img_bgr=None) -> np.ndarray:
    """Depth for an image file as float32 [0, 1] at the image's resolution.

    Uses the bundled map for reference samples, then the cache, and only runs the model when needed.
    """
    image_path = Path(image_path)
    if img_bgr is None:
        img_bgr = cv2.imread(str(image_path))
    h, w = img_bgr.shape[:2]

    candidates = []
    if image_path.resolve().parent == SAMPLES_DIR.resolve():
        candidates.append(DEPTH_DIR / f"{image_path.stem}.png")
    cached = DEPTH_CACHE_DIR / f"{file_hash(image_path)}.png"
    candidates.append(cached)

    for c in candidates:
        if c.exists():
            d = cv2.imread(str(c), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
            return cv2.resize(d, (w, h), interpolation=cv2.INTER_LINEAR) if d.shape != (h, w) else d

    print(f"[depth] Estimating depth for {image_path.name} (first time only)...")
    d = estimate_depth(img_bgr)
    if d.shape != (h, w):
        d = cv2.resize(d, (w, h), interpolation=cv2.INTER_LINEAR)
    DEPTH_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(cached), (d * 255.0).round().astype(np.uint8))
    return d
