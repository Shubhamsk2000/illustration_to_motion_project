"""Render a list of timed shots into one video, with crossfades, one cached segment per shot.

Each shot's motion runs from half a crossfade before its cut to half a crossfade after, so the fade blends two
moving pictures. Segment i covers exactly [start_i, end_i): its first half-fade blends in the previous shot, its
last half-fade blends towards the next one, so segments join seamlessly and are concatenated without re-encoding.
A segment is only re-rendered when something that affects it (its image, look or timing, or a neighbour's)
changes.
"""

import hashlib
import json
import subprocess
import textwrap
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from motion_studio.depth import file_hash
from motion_studio.encode import VideoWriter
from motion_studio.scene import SceneRenderer, SceneSpec

ENGINE_VERSION = "2"


@dataclass
class Shot:
    id: str
    image: Path
    spec: SceneSpec
    start: float
    end: float
    placeholder: bool = False


def make_placeholder(path: Path, shot_id: str, text: str, size=(1920, 1080)) -> None:
    """A dark card with the scene number and narration, used until the real image exists."""
    w, h = size
    yy = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
    img = (np.array([40, 30, 25], np.float32) * (1 - yy) + np.array([70, 55, 45], np.float32) * yy)
    img = np.repeat(img, w, axis=1).astype(np.uint8)
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(img, f"Scene {shot_id}", (80, 170), font, 2.6, (200, 220, 235), 3, cv2.LINE_AA)
    cv2.putText(img, f"image missing: add images/{shot_id}.png", (80, 240), font, 1.1, (120, 170, 230), 2, cv2.LINE_AA)
    for k, line in enumerate(textwrap.wrap(text, 58)[:9]):
        cv2.putText(img, line, (80, 360 + k * 70), font, 1.4, (235, 235, 235), 2, cv2.LINE_AA)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)


def _flat_depth(path: Path) -> np.ndarray:
    h, w = cv2.imread(str(path)).shape[:2]
    return np.repeat(np.linspace(0.2, 0.5, h, dtype=np.float32)[:, None], w, axis=1)


class _Renderers:
    """Keeps only the renderers needed for the current shot and its neighbours."""

    def __init__(self, shots, size, fps, transition):
        self.shots, self.size, self.fps, self.f = shots, size, fps, transition
        self.live: dict[int, SceneRenderer] = {}

    def window(self, i: int) -> tuple[float, float]:
        s = self.shots[i]
        m0 = s.start - (self.f / 2 if i > 0 else 0.0)
        m1 = s.end + (self.f / 2 if i < len(self.shots) - 1 else 0.0)
        return m0, m1

    def get(self, i: int) -> SceneRenderer:
        if i not in self.live:
            s = self.shots[i]
            m0, m1 = self.window(i)
            depth = _flat_depth(s.image) if s.placeholder else None
            self.live[i] = SceneRenderer(s.image, s.spec, size=self.size, fps=self.fps, duration=m1 - m0, depth=depth)
        for k in [k for k in self.live if abs(k - i) > 1]:
            del self.live[k]
        return self.live[i]

    def frame(self, i: int, T: float) -> np.ndarray:
        m0, m1 = self.window(i)
        return self.get(i).frame_at((T - m0) / (m1 - m0))


def _segment_key(shots, i, size, fps, f, encoder) -> str:
    parts = {"v": ENGINE_VERSION, "size": size, "fps": fps, "f": f, "enc": encoder}
    for k in (i - 1, i, i + 1):
        if 0 <= k < len(shots):
            s = shots[k]
            parts[str(k - i)] = [file_hash(s.image), asdict(s.spec), s.start, s.end, s.placeholder]
    return hashlib.sha1(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:12]


def render_timeline(shots: list[Shot], out_video: Path, build_dir: Path, size=(1920, 1080), fps: int = 24,
                    transition: float = 0.6, draft: bool = False) -> dict:
    seg_dir = build_dir / "segments"
    seg_dir.mkdir(parents=True, exist_ok=True)
    f = max(0.0, min(transition, min(s.end - s.start for s in shots)))
    crf, enc = (23, "ultrafast") if draft else (18, "veryfast")
    rs = _Renderers(shots, size, fps, f)
    last = len(shots) - 1
    seg_paths, rendered, reused = [], 0, 0

    for i, s in enumerate(shots):
        key = _segment_key(shots, i, size, fps, f, enc)
        path = seg_dir / f"{s.id}_{key}.mp4"
        seg_paths.append(path)
        for old in seg_dir.glob(f"{s.id}_*.mp4"):
            if old != path:
                old.unlink()
        if path.exists():
            reused += 1
            continue

        g0, g1 = round(s.start * fps), round(s.end * fps)
        with VideoWriter(path, size[0], size[1], fps, crf=crf, preset=enc) as vw:
            for g in range(g0, g1):
                T = g / fps
                frame = rs.frame(i, T)
                if f > 0 and i > 0 and T < s.start + f / 2:
                    w = 0.5 + (T - s.start) / f
                    frame = cv2.addWeighted(frame, w, rs.frame(i - 1, T), 1 - w, 0)
                elif f > 0 and i < last and T >= s.end - f / 2:
                    w = 0.5 + (s.end - T) / f
                    frame = cv2.addWeighted(frame, w, rs.frame(i + 1, T), 1 - w, 0)
                vw.write(frame)
        rendered += 1
        print(f"\r[render] scene {s.id} ({i + 1}/{len(shots)})", end="", flush=True)
    print()

    listing = build_dir / "segments.txt"
    listing.write_text("".join(f"file '{p.resolve()}'\n" for p in seg_paths))
    out_video.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(listing),
                    "-c", "copy", str(out_video)], check=True)
    return {"rendered": rendered, "reused": reused}
