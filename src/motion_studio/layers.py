"""Split an illustration into flat 2D planes: background (with subjects filled in), subject, foreground."""

from dataclasses import dataclass

import cv2
import numpy as np

# Inpainting runs at this size and is upscaled: much faster, and the fill is soft either way.
_INPAINT_MAX_SIDE = 640


@dataclass
class Layers:
    image: np.ndarray            # BGR uint8, H x W
    background: np.ndarray       # BGR uint8, subjects removed and filled in
    mid_alpha: np.ndarray        # float32 [0, 1], the subject plane
    fg_alpha: np.ndarray | None  # float32 [0, 1], an occluding foreground plane if one was found
    anchor: tuple[float, float]  # (x, y) at the subject's base (feet), for breathing
    center: tuple[float, float]  # (x, y) centroid of the subject


def _fill_behind(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Fill the masked area (uint8 0/255) from its surroundings."""
    h, w = img.shape[:2]
    s = min(1.0, _INPAINT_MAX_SIDE / max(h, w))
    small = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    small_mask = cv2.resize(mask, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST)
    filled = cv2.inpaint(small, small_mask, max(3, int(9 * s)), cv2.INPAINT_TELEA)
    filled = cv2.resize(filled, (w, h), interpolation=cv2.INTER_CUBIC)
    soft = cv2.GaussianBlur(mask, (0, 0), 3).astype(np.float32)[:, :, None] / 255.0
    return (img * (1.0 - soft) + filled * soft).astype(np.uint8)


def split_layers(img: np.ndarray, depth01: np.ndarray) -> Layers:
    """Depth-threshold layer split (from approach 4), at the image's own resolution."""
    h, w = img.shape[:2]
    k = max(1, round(max(h, w) / 900))  # scale kernel sizes with resolution (tuned at ~720-900 px)

    def odd(n):
        n = int(n)
        return n if n % 2 == 1 else n + 1

    depth8 = (np.clip(depth01, 0, 1) * 255).astype(np.uint8)
    if depth8.shape != (h, w):
        depth8 = cv2.resize(depth8, (w, h), interpolation=cv2.INTER_LINEAR)
    depth_bi = cv2.bilateralFilter(depth8, d=9, sigmaColor=75, sigmaSpace=75)
    t_split, _ = cv2.threshold(depth_bi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Is there a separate occluding foreground (e.g. grass in front of the subject)?
    fg_depths = depth_bi[depth_bi >= t_split]
    t_fg = None
    if fg_depths.size and (np.percentile(fg_depths, 95) - np.percentile(fg_depths, 20)) > 70:
        cand_t = int(np.percentile(fg_depths, 75))
        sobel = np.abs(cv2.Sobel(depth_bi, cv2.CV_32F, 1, 1, ksize=3))
        edge_band = np.abs(depth_bi.astype(np.float32) - cand_t) < 8
        if edge_band.any() and sobel[edge_band].mean() > 15.0:
            t_fg = cand_t

    open_k = np.ones((5 * k, 5 * k), np.uint8)
    if t_fg is not None:
        fg_mask = cv2.morphologyEx((depth_bi >= t_fg).astype(np.uint8) * 255, cv2.MORPH_OPEN, open_k)
        fg_alpha = cv2.GaussianBlur(fg_mask, (odd(5 * k), odd(5 * k)), 0).astype(np.float32) / 255.0
        mid_mask = ((depth_bi >= t_split) & (depth_bi < t_fg)).astype(np.uint8) * 255
        mid_mask = cv2.morphologyEx(mid_mask, cv2.MORPH_OPEN, open_k)
        blur = odd(5 * k)
    else:
        fg_alpha = None
        mid_mask = (depth_bi >= t_split).astype(np.uint8) * 255
        mid_mask = cv2.morphologyEx(mid_mask, cv2.MORPH_CLOSE, np.ones((7 * k, 7 * k), np.uint8))
        blur = odd(7 * k)
    mid_alpha = cv2.GaussianBlur(mid_mask, (blur, blur), 0).astype(np.float32) / 255.0

    occluders = (depth_bi >= t_split).astype(np.uint8) * 255
    occluders = cv2.dilate(occluders, np.ones((25 * k, 25 * k), np.uint8))
    background = _fill_behind(img, occluders)

    pts = cv2.findNonZero(mid_mask)
    if pts is not None:
        pts = pts.reshape(-1, 2).astype(np.float32)
        anchor = (float(pts[:, 0].mean()), float(np.percentile(pts[:, 1], 85)))
        center = (float(pts[:, 0].mean()), float(pts[:, 1].mean()))
    else:
        anchor, center = (w * 0.5, h * 0.75), (w * 0.5, h * 0.5)

    return Layers(img, background, mid_alpha, fg_alpha, anchor, center)
