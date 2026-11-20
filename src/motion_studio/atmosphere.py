"""Atmospheric particles (dust, embers, fireflies, snow, rain), computed from time so any frame can be rendered alone."""

import math

import cv2
import numpy as np

# speed: vertical speed in frame-heights per second (negative = rising). sway: horizontal sway in px at 1080p.
KINDS = {
    "dust":      dict(color=(200, 230, 255), size=(1.5, 3.5), speed=(-0.04, -0.015), sway=(6, 16), flicker=0.35, count=40),
    "embers":    dict(color=(60, 150, 255), size=(1.2, 2.8), speed=(-0.18, -0.08), sway=(8, 22), flicker=0.7, count=45),
    "fireflies": dict(color=(120, 255, 190), size=(2.0, 3.5), speed=(-0.01, 0.01), sway=(15, 30), flicker=0.85, count=25),
    "snow":      dict(color=(255, 250, 245), size=(1.5, 3.5), speed=(0.06, 0.14), sway=(10, 25), flicker=0.1, count=90),
    "rain":      dict(color=(230, 215, 200), size=(1.0, 1.0), speed=(1.2, 1.8), sway=(0, 0), flicker=0.2, count=110),
}
KIND_NAMES = ["none"] + list(KINDS)

_SPRITES: dict[float, np.ndarray] = {}


def _sprite(radius: float) -> np.ndarray:
    """Soft round glow, float32 [0, 1], cached per radius."""
    r = round(radius * 2) / 2
    if r not in _SPRITES:
        half = int(math.ceil(r * 2.5))
        yy, xx = np.mgrid[-half:half + 1, -half:half + 1].astype(np.float32)
        sigma = r * 0.9
        _SPRITES[r] = np.exp(-0.5 * (xx * xx + yy * yy) / (sigma * sigma))
    return _SPRITES[r]


class Particles:
    def __init__(self, kind: str, width: int, height: int, duration: float, loop: bool,
                 count: int | None = None, seed: int = 7):
        cfg = KINDS[kind]
        self.kind, self.w, self.h = kind, width, height
        self.color = np.array(cfg["color"], np.float32)
        scale = height / 1080.0
        n = int(count if count is not None else cfg["count"])
        rng = np.random.default_rng(seed)

        self.margin = 20.0 * scale
        span = height + 2 * self.margin
        self.x0 = rng.uniform(0, width, n)
        self.y0 = rng.uniform(0, span, n)
        self.size = rng.uniform(*cfg["size"], n) * max(scale, 0.5)
        self.sway = rng.uniform(*cfg["sway"], n) * scale
        self.phase = rng.uniform(0, 2 * math.pi, n)
        self.phase2 = rng.uniform(0, 2 * math.pi, n)
        travel = rng.uniform(*cfg["speed"], n) * height * duration  # px over the whole clip
        if loop:  # whole number of trips through the frame, so the last frame meets the first
            travel = np.round(travel / span) * span
        self.travel = travel
        self.span = span
        self.flicker = cfg["flicker"]
        cycles = max(1, round(duration / 4.0))
        self.sway_cycles = cycles
        self.flicker_cycles = max(1, round(duration / 1.5)) if kind in ("embers", "fireflies") else cycles

    def draw(self, frame: np.ndarray, t: float, shift: tuple[float, float] = (0.0, 0.0)) -> None:
        """Draw onto a BGR uint8 frame in place. shift moves particles with the camera (parallax)."""
        tau = 2 * math.pi * t
        y = (self.y0 + self.travel * t) % self.span - self.margin + shift[1]
        x = self.x0 + self.sway * np.sin(self.sway_cycles * tau + self.phase) + shift[0]
        x = x % self.w
        fade_edge = np.clip(np.sin(np.pi * np.clip(y / self.h, 0, 1)) * 3.0, 0, 1)
        flick = 1.0 - self.flicker * (0.5 + 0.5 * np.sin(self.flicker_cycles * tau + self.phase2))
        alpha = fade_edge * flick

        h, w = frame.shape[:2]
        if self.kind == "rain":
            length = 0.012 * self.h
            for xi, yi, a in zip(x, y, alpha):
                if a > 0.02:
                    c = tuple(float(v) * a * 0.8 for v in self.color)
                    cv2.line(frame, (int(xi), int(yi)), (int(xi - length * 0.2), int(yi + length)), c, 1, cv2.LINE_AA)
            return

        for xi, yi, si, a in zip(x, y, self.size, alpha):
            if a <= 0.02:
                continue
            spr = _sprite(si)
            half = spr.shape[0] // 2
            ix, iy = int(round(xi)), int(round(yi))
            x1, x2 = max(0, ix - half), min(w, ix + half + 1)
            y1, y2 = max(0, iy - half), min(h, iy + half + 1)
            if x2 <= x1 or y2 <= y1:
                continue
            patch = spr[y1 - iy + half:y2 - iy + half, x1 - ix + half:x2 - ix + half]
            glow = (patch[:, :, None] * (self.color * a)).astype(np.uint8)
            frame[y1:y2, x1:x2] = cv2.add(frame[y1:y2, x1:x2], glow)
