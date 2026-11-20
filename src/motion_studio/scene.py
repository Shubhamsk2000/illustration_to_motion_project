"""Scene engine: one illustration -> one 16:9 video clip of any length.

The picture is split into flat planes (background, subject, foreground). A camera move is applied to every plane
with a different strength, so near things move more than far things (2.5D parallax) while the art itself is never
warped. On top: subject breathing, a light pulse and atmospheric particles.
"""

import math
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import cv2
import numpy as np

from motion_studio.atmosphere import Particles
from motion_studio.camera import camera_at
from motion_studio.depth import depth_for_image, file_hash
from motion_studio.effects import Glow, Mist
from motion_studio.encode import VideoWriter
from motion_studio.layers import Layers, split_layers
from motion_studio.paths import OUTPUT_DIR

LAYER_CACHE_DIR = OUTPUT_DIR / "cache" / "layers"
LAYERS_VERSION = "v2"  # bump when layer splitting or the fill changes, so caches are rebuilt
FITS = ["auto", "crop", "blur"]


@dataclass
class SceneSpec:
    move: str = "push_in"         # see camera.ALL_MOVES
    strength: float = 1.0         # size of the camera move
    depth: float = 1.0            # how strongly near and far planes separate
    breathing: float = 0.006      # subject breathing (fraction of height); 0 = off
    atmosphere: str = "none"      # see atmosphere.KIND_NAMES
    particle_count: int | None = None
    light: float = 0.03           # warm light pulse; 0 = off
    mist: str = "none"            # fog (drifts behind the subject) | smoke (rises over everything)
    glow: str = "none"            # rays (light shafts from the brightest spot) | firelight (flicker around fire)
    loop: bool = False            # seamless loop instead of a one-way move
    fit: str = "auto"             # auto | crop (fill 16:9) | blur (whole image on a blurred backdrop)


PRESETS: dict[str, dict] = {
    "documentary": dict(move="push_in", atmosphere="dust"),
    "portrait": dict(move="push_in", strength=0.6, breathing=0.008),
    "epic": dict(move="push_in", strength=1.6, depth=1.4, atmosphere="dust", light=0.05),
    "calm": dict(move="drift", strength=0.6, light=0.02),
    "campfire": dict(move="push_in", strength=0.7, atmosphere="embers", glow="firelight", light=0.04),
    "winter": dict(move="pan_right", atmosphere="snow", light=0.0),
    "storm": dict(move="push_in", strength=0.8, atmosphere="rain", light=0.0),
    "magic_forest": dict(move="drift", atmosphere="fireflies", light=0.04),
    "misty": dict(move="push_in", strength=0.7, mist="fog", glow="rays", light=0.02),
    "sunlit": dict(move="push_in", atmosphere="dust", glow="rays", light=0.03),
    "battle": dict(move="push_in", strength=1.4, depth=1.3, atmosphere="embers", mist="smoke", light=0.05),
}


def spec_from_preset(preset: str | None, **overrides) -> SceneSpec:
    base = SceneSpec()
    if preset:
        if preset not in PRESETS:
            raise ValueError(f"Unknown preset '{preset}'. Choose from: {', '.join(PRESETS)}")
        base = replace(base, **PRESETS[preset])
    return replace(base, **{k: v for k, v in overrides.items() if v is not None})


def choose_fit(fit: str, image_aspect: float, frame_aspect: float) -> str:
    if fit != "auto":
        return fit
    # Crop when little would be lost (e.g. 3:2 or wider); otherwise show it all on a blurred backdrop.
    return "crop" if image_aspect >= frame_aspect * 0.78 else "blur"


def _load_layers(image_path: Path, plate: np.ndarray, depth: np.ndarray | None) -> Layers:
    ph, pw = plate.shape[:2]
    key = f"{file_hash(image_path)}_{pw}x{ph}_{LAYERS_VERSION}"
    cache = LAYER_CACHE_DIR / f"{key}.npz"
    if depth is None and cache.exists():
        z = np.load(cache)
        fg = z["fg_alpha"].astype(np.float32) / 255.0 if z["has_fg"] else None
        return Layers(plate, z["background"], z["mid_alpha"].astype(np.float32) / 255.0, fg,
                      tuple(z["anchor"]), tuple(z["center"]))

    if depth is None:
        depth = depth_for_image(image_path)
    depth = cv2.resize(depth.astype(np.float32), (pw, ph), interpolation=cv2.INTER_LINEAR)
    layers = split_layers(plate, depth)

    LAYER_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    to8 = lambda a: (a * 255).round().astype(np.uint8)  # noqa: E731
    np.savez_compressed(
        cache, background=layers.background, mid_alpha=to8(layers.mid_alpha),
        fg_alpha=to8(layers.fg_alpha) if layers.fg_alpha is not None else np.zeros((1, 1), np.uint8),
        has_fg=layers.fg_alpha is not None, anchor=np.array(layers.anchor), center=np.array(layers.center),
    )
    return layers


def _affine(m2x3: np.ndarray) -> np.ndarray:
    return np.vstack([m2x3, [0.0, 0.0, 1.0]])


class SceneRenderer:
    def __init__(self, image_path, spec: SceneSpec | None = None, size=(1920, 1080), fps: float = 24,
                 duration: float = 6.0, depth: np.ndarray | None = None):
        self.image_path = Path(image_path)
        self.spec = spec or SceneSpec()
        self.W, self.H = (int(size[0]) // 2 * 2, int(size[1]) // 2 * 2)
        self.fps, self.duration = fps, float(duration)
        self.n_frames = max(2, round(self.duration * fps))

        img = cv2.imread(str(self.image_path), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Could not read image: {self.image_path}")
        h, w = img.shape[:2]
        self.fit = choose_fit(self.spec.fit, w / h, self.W / self.H)

        # Plate = the image resized to cover the area it is shown in (the whole frame, or a centred column).
        if self.fit == "crop":
            s = max(self.W / w, self.H / h)
            self.ow, self.oh = self.W, self.H
        else:
            s = self.H / h
            self.ow, self.oh = min(self.W, int(w * s) // 2 * 2), self.H
        pw, ph = max(self.ow, math.ceil(w * s)), max(self.oh, math.ceil(h * s))
        interp = cv2.INTER_AREA if s < 1 else cv2.INTER_LANCZOS4
        plate = cv2.resize(img, (pw, ph), interpolation=interp)

        self.layers = _load_layers(self.image_path, plate, depth)
        self._setup_camera(pw, ph)
        self._setup_backdrop(img)

        sp = self.spec
        self.particles = None
        if sp.atmosphere and sp.atmosphere != "none":
            self.particles = Particles(sp.atmosphere, self.ow, self.oh, self.duration, sp.loop, sp.particle_count)
        self.mist = None
        if sp.mist and sp.mist != "none":
            self.mist = Mist(sp.mist, self.ow, self.oh, self.duration, sp.loop)
        self.glow = None
        if sp.glow and sp.glow != "none":
            self.glow = Glow(sp.glow, self.layers.image, self.duration)
        self.breath_cycles = max(1, round(self.duration / 3.5))
        self.light_cycles = max(1, round(self.duration / 3.5))
        no_fire_found = sp.glow == "firelight" and self.glow.map is None
        # Fire with no visible flame: flicker the whole picture instead.
        flicker = sp.atmosphere == "embers" or no_fire_found
        self.flicker_cycles = max(2, round(self.duration * 1.3)) if flicker else 0
        self._lut_base = np.arange(256, dtype=np.float32)

    # ---- setup -------------------------------------------------------------------------------------------

    def _setup_camera(self, pw: int, ph: int) -> None:
        sp, L = self.spec, self.layers
        self.p_bg = 1.0 - 0.3 * sp.depth
        self.p_fg = 1.0 + 0.35 * sp.depth
        self.p_particles = 1.0 + 0.5 * sp.depth

        # Frame the subject: centre the visible window on it, as far as the plate allows.
        self.wcx = float(np.clip(L.center[0], self.ow / 2, pw - self.ow / 2))
        self.wcy = float(np.clip(L.center[1], self.oh / 2, ph - self.oh / 2))

        # Extra zoom so the background plane always covers the frame for the whole move.
        samples = [camera_at(sp.move, t, sp.strength) for t in np.linspace(0, 1, 61)]
        max_dx = max(abs(c[1]) for c in samples) * self.ow
        max_dy = max(abs(c[2]) for c in samples) * self.oh
        room_x = min(self.wcx, pw - self.wcx)
        room_y = min(self.wcy, ph - self.wcy)
        z0 = 1.0
        for p in (self.p_bg, 1.0):
            z0 = max(z0, (self.ow / 2 + max_dx * p) / room_x, (self.oh / 2 + max_dy * p) / room_y)
        self.z0 = z0 * 1.002

    def _setup_backdrop(self, img: np.ndarray) -> None:
        self.backdrop, self.x_off = None, 0
        if self.fit != "blur" or self.ow >= self.W:
            return
        h, w = img.shape[:2]
        s = max(self.W / w, self.H / h) / 8.0  # blur at 1/8 size: fast and smooth
        small = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(small, (0, 0), 6)
        cw, ch = max(self.W, round(w * s * 8)), max(self.H, round(h * s * 8))
        big = cv2.resize(small, (cw, ch), interpolation=cv2.INTER_CUBIC)
        y0, x0 = (big.shape[0] - self.H) // 2, (big.shape[1] - self.W) // 2
        big = big[y0:y0 + self.H, x0:x0 + self.W].astype(np.float32)

        self.x_off = (self.W - self.ow) // 2
        xs = np.arange(self.W, dtype=np.float32)
        dist = np.maximum(self.x_off - xs, xs - (self.x_off + self.ow))  # distance from the picture's edge
        shade = 0.45 * (1.0 - 0.45 * np.exp(-np.maximum(dist, 0) / (0.02 * self.W)))
        self.backdrop = (big * shade[None, :, None]).astype(np.uint8)

    # ---- per frame ---------------------------------------------------------------------------------------

    def _matrix(self, p: float, zoom: float, dx: float, dy: float) -> np.ndarray:
        Z = self.z0 * (1.0 + (zoom - 1.0) * p)
        return np.array([
            [Z, 0.0, self.ow / 2 + dx * self.ow * p - self.wcx * Z],
            [0.0, Z, self.oh / 2 + dy * self.oh * p - self.wcy * Z],
        ])

    def _warp(self, src: np.ndarray, m: np.ndarray) -> np.ndarray:
        return cv2.warpAffine(src, m, (self.ow, self.oh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    def _light_lut(self, t: float) -> np.ndarray | None:
        amt = self.spec.light
        if amt <= 0:
            return None
        s = 0.5 + 0.5 * math.sin(2 * math.pi * self.light_cycles * t)
        if self.flicker_cycles:  # fire light: add a quicker, irregular flicker
            s = 0.5 * s + 0.3 * (0.5 + 0.5 * math.sin(2 * math.pi * self.flicker_cycles * t)) \
                + 0.2 * (0.5 + 0.5 * math.sin(2 * math.pi * (self.flicker_cycles + 3) * t + 1.3))
        b = self._lut_base
        lut = np.stack([b, b * (1 + 0.4 * amt * s), b * (1 + amt * s)], axis=-1)
        return np.clip(lut, 0, 255).astype(np.uint8).reshape(1, 256, 3)

    def frame_at(self, t: float) -> np.ndarray:
        """Render the frame at normalized time t in [0, 1] as BGR uint8 (W x H)."""
        sp, L = self.spec, self.layers
        zoom, dx, dy = camera_at(sp.move, t, sp.strength)

        m_bg = self._matrix(self.p_bg, zoom, dx, dy)
        frame = self._warp(L.background, m_bg)
        bg_shift = (dx * self.ow * self.p_bg, dy * self.oh * self.p_bg)
        if self.mist is not None and self.mist.behind_subject:
            frame = self.mist.apply(frame, t, bg_shift)

        m_mid = self._matrix(1.0, zoom, dx, dy)
        if sp.breathing > 0:
            b = sp.breathing * math.sin(2 * math.pi * self.breath_cycles * t)
            sy, sx = 1.0 + b, 1.0 + 0.35 * b
            ax, ay = L.anchor
            breath = np.array([[sx, 0.0, (1 - sx) * ax], [0.0, sy, (1 - sy) * ay]])
            m_mid = (_affine(m_mid) @ _affine(breath))[:2]
        a = self._warp(L.mid_alpha, m_mid)
        frame = cv2.blendLinear(self._warp(L.image, m_mid), frame, a, 1.0 - a)

        if L.fg_alpha is not None:
            m_fg = self._matrix(self.p_fg, zoom, dx, dy)
            a = self._warp(L.fg_alpha, m_fg)
            frame = cv2.blendLinear(self._warp(L.image, m_fg), frame, a, 1.0 - a)

        if self.glow is not None:
            frame = self.glow.apply(frame, t, lambda m: self._warp(m, m_bg))
        if self.mist is not None and not self.mist.behind_subject:
            frame = self.mist.apply(frame, t, bg_shift)

        if self.particles is not None:
            shift = (dx * self.ow * self.p_particles, dy * self.oh * self.p_particles)
            self.particles.draw(frame, t, shift)

        lut = self._light_lut(t)
        if lut is not None:
            frame = cv2.LUT(frame, lut)

        if self.backdrop is not None:
            out = self.backdrop.copy()
            out[:, self.x_off:self.x_off + self.ow] = frame
            return out
        return frame

    def times(self) -> list[float]:
        n = self.n_frames
        # A loop never repeats its first frame at the end; a one-way move ends exactly on its final pose.
        return [i / n for i in range(n)] if self.spec.loop else [i / (n - 1) for i in range(n)]

    def render(self, out_path, progress=None, crf: int = 18, encoder_preset: str = "veryfast") -> dict:
        t0 = time.time()
        with VideoWriter(out_path, self.W, self.H, self.fps, crf=crf, preset=encoder_preset) as vw:
            ts = self.times()
            for i, t in enumerate(ts):
                vw.write(self.frame_at(t))
                if progress and (i % 12 == 0 or i == len(ts) - 1):
                    progress((i + 1) / len(ts))
        took = time.time() - t0
        notes = []
        if self.glow is not None and self.glow.map is None:
            notes.append("no fire found, flickering the whole picture" if self.spec.glow == "firelight"
                         else "no bright light source found, so no rays")
        return {"path": str(out_path), "frames": self.n_frames, "seconds": self.duration, "fit": self.fit,
                "notes": notes,
                "render_seconds": round(took, 2), "spec": asdict(self.spec)}


def render_scene(image_path, out_path, spec: SceneSpec | None = None, duration: float = 6.0,
                 size=(1920, 1080), fps: float = 24, progress=None) -> dict:
    return SceneRenderer(image_path, spec, size=size, fps=fps, duration=duration).render(out_path, progress)
