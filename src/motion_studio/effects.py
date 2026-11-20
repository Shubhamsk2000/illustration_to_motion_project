"""Scene effects on top of the planes: drifting mist (fog, smoke) and light (rays, firelight).

Everything is precomputed once per scene and only shifted / scaled per frame, so effects cost a few
milliseconds per frame. Any frame can be rendered on its own from its time t in [0, 1].
"""

import math

import cv2
import numpy as np

MIST_KINDS = {
    # speed: drift in frame-widths (x) / frame-heights (y) per second; cell: noise feature size as a frame fraction
    "fog":   dict(color=(232, 228, 222), opacity=0.42, speed=(0.012, 0.0), cell=0.22, behind_subject=True),
    "smoke": dict(color=(150, 155, 160), opacity=0.30, speed=(0.004, -0.02), cell=0.12, behind_subject=False),
}
MIST_NAMES = ["none"] + list(MIST_KINDS)
GLOW_NAMES = ["none", "rays", "firelight"]


def _tileable_noise(h: int, w: int, cell: float, rng: np.random.Generator) -> np.ndarray:
    """Smooth noise in [0, 1] that wraps seamlessly in both directions (two octaves)."""
    out = np.zeros((h, w), np.float32)
    for octave, weight in ((1, 0.65), (2, 0.35)):
        gw = max(2, round(1 / (cell / octave)))
        gh = max(2, round(gw * h / w))
        grid = rng.random((gh, gw)).astype(np.float32)
        pad = np.pad(grid, 2, mode="wrap")  # wrap so the upscaled field tiles
        sx, sy = w / gw, h / gh
        big = cv2.resize(pad, (int(round(pad.shape[1] * sx)), int(round(pad.shape[0] * sy))),
                         interpolation=cv2.INTER_CUBIC)
        y0, x0 = int(round(2 * sy)), int(round(2 * sx))
        out += weight * big[y0:y0 + h, x0:x0 + w]
    out -= out.min()
    return out / max(1e-6, out.max())


class Mist:
    def __init__(self, kind: str, width: int, height: int, duration: float, loop: bool, amount: float = 1.0,
                 seed: int = 11):
        cfg = MIST_KINDS[kind]
        self.kind, self.w, self.h = kind, width, height
        self.behind_subject = cfg["behind_subject"]
        rng = np.random.default_rng(seed)
        noise = _tileable_noise(height, width, cfg["cell"], rng)
        self.tex = np.clip((noise - 0.3) / 0.6, 0, 1).astype(np.float32)  # patches with clear gaps

        if kind == "fog":  # thicker towards the ground, thin in the sky
            weight = np.clip(0.25 + 0.95 * np.linspace(0, 1, height, dtype=np.float32), 0, 1)[:, None]
        else:
            weight = np.ones((height, 1), np.float32)
        self.weight = (weight * cfg["opacity"] * amount).astype(np.float32)
        self.color = np.empty((height, width, 3), np.uint8)
        self.color[:] = cfg["color"]

        tx, ty = cfg["speed"][0] * width * duration, cfg["speed"][1] * height * duration
        if loop:  # whole tiles only, so the last frame meets the first
            tx, ty = round(tx / width) * width, round(ty / height) * height
        self.travel = (tx, ty)

    def alpha(self, t: float, shift=(0.0, 0.0)) -> np.ndarray:
        layers = []
        for speed, offset in ((1.0, 0.0), (-0.6, 0.37)):  # two sheets drifting differently look alive
            dx = self.travel[0] * t * speed + offset * self.w + shift[0]
            dy = self.travel[1] * t * abs(speed) + shift[1]
            m = np.float32([[1, 0, dx], [0, 1, dy]])
            layers.append(cv2.warpAffine(self.tex, m, (self.w, self.h), flags=cv2.INTER_LINEAR,
                                         borderMode=cv2.BORDER_WRAP))
        return (0.5 * (layers[0] + layers[1])) * self.weight

    def apply(self, frame: np.ndarray, t: float, shift=(0.0, 0.0)) -> np.ndarray:
        a = self.alpha(t, shift)
        return cv2.blendLinear(self.color, frame, a, 1.0 - a)


# ---- light --------------------------------------------------------------------------------------------------

def find_light_source(img: np.ndarray) -> tuple[float, float] | None:
    """The brightest broad spot in the picture, if it is bright enough to cast rays."""
    h, w = img.shape[:2]
    lum = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    blur = cv2.GaussianBlur(lum, (0, 0), max(h, w) * 0.03)
    _, peak, _, loc = cv2.minMaxLoc(blur)
    return (float(loc[0]), float(loc[1])) if peak >= 200 else None


def find_fire(img: np.ndarray) -> tuple[float, float] | None:
    """Centre of the brightest warm (orange/yellow) area - flames are brighter than skin or cloth."""
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (5, 90, 170), (35, 255, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, labels, stats, cents = cv2.connectedComponentsWithStats(mask)
    if n <= 1:
        return None
    v = hsv[:, :, 2].astype(np.float64) / 255.0
    best, best_score = None, 0.0
    for k in range(1, n):
        if stats[k, cv2.CC_STAT_AREA] < 0.0005 * h * w:
            continue
        vals = v[labels == k]
        score = float(np.percentile(vals, 90)) * 4 + float(np.mean(vals))  # brightness first, then size of glow
        if score > best_score:
            best, best_score = (float(cents[k][0]), float(cents[k][1])), score
    return best


class Glow:
    """A precomputed light map (plate coordinates) added to the frame with a time-varying strength."""

    def __init__(self, kind: str, plate: np.ndarray, duration: float, amount: float = 1.0):
        self.kind = kind
        h, w = plate.shape[:2]
        diag = math.hypot(w, h)
        self.map = None
        self.source = find_light_source(plate) if kind == "rays" else find_fire(plate)
        if self.source is None:
            return
        sx, sy = self.source
        if kind == "rays":
            lum = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY).astype(np.float32)
            bright = np.clip((lum - 170) / 85, 0, 1)
            small_s = 480 / max(h, w)  # build the streaks at low resolution: they are soft anyway
            sm = cv2.resize(bright, (int(w * small_s), int(h * small_s)), interpolation=cv2.INTER_AREA)
            cx, cy = sx * small_s, sy * small_s
            acc = np.zeros_like(sm)
            scales = np.linspace(1.0, 1.9, 24)
            for s in scales:  # radial blur away from the source = light shafts
                m = np.float32([[s, 0, (1 - s) * cx], [0, s, (1 - s) * cy]])
                acc += cv2.warpAffine(sm, m, (sm.shape[1], sm.shape[0]), flags=cv2.INTER_LINEAR)
            acc /= len(scales)
            rays = cv2.resize(cv2.GaussianBlur(acc, (0, 0), 2), (w, h), interpolation=cv2.INTER_CUBIC)
            rays *= (1.0 - lum / 255.0) ** 2  # shafts show where they cross darker areas; bright sky can't clip
            color = np.array([200, 235, 255], np.float32)
            strength = 0.9 * amount
        else:
            yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
            d = np.sqrt((xx - sx) ** 2 + (yy - sy) ** 2) / (0.35 * diag)
            rays = np.exp(-d * d * 2.5)
            color = np.array([40, 140, 255], np.float32)
            strength = 0.45 * amount
        self.map = np.clip(rays[:, :, None] * color * strength, 0, 255).astype(np.uint8)
        self.cycles = max(1, round(duration / 4.0))
        self.flicker = (max(3, round(duration * 1.6)), max(5, round(duration * 2.7)))

    def intensity(self, t: float) -> float:
        tau = 2 * math.pi * t
        if self.kind == "rays":  # slow breathing of the light
            return 0.75 + 0.25 * math.sin(self.cycles * tau)
        a, b = self.flicker  # fire: irregular flicker from two quick waves
        return 0.6 + 0.25 * math.sin(a * tau) + 0.15 * math.sin(b * tau + 1.7)

    def apply(self, frame: np.ndarray, t: float, warp) -> np.ndarray:
        if self.map is None:
            return frame
        return cv2.addWeighted(frame, 1.0, warp(self.map), self.intensity(t), 0)
