"""Quick low-resolution preview of a single scene with its current look."""

import hashlib
import json
from dataclasses import asdict

from motion_studio.depth import file_hash
from motion_studio.episode.build import placeholder_image, scene_specs
from motion_studio.episode.project import Episode
from motion_studio.episode.script import WORDS_PER_MINUTE
from motion_studio.scene import SceneRenderer

PREVIEW_SIZE = (640, 360)


def scene_duration(ep: Episode, scene) -> float:
    """The scene's length from the last build, or an estimate from its word count."""
    tl = ep.export_dir / "timeline.json"
    if tl.exists():
        for item in json.loads(tl.read_text()):
            if item["id"] == scene.id:
                return round(item["end"] - item["start"], 2)
    return round(min(14.0, max(3.0, len(scene.words) * 60 / WORDS_PER_MINUTE)), 2)


def preview_scene(ep: Episode, scene_id: str) -> dict:
    items = {sc["scene"].id: sc for sc in scene_specs(ep)}
    if scene_id not in items:
        raise KeyError(scene_id)
    sc = items[scene_id]
    scene, spec = sc["scene"], sc["spec"]
    image = ep.images().get(int(scene.id))
    placeholder = image is None
    if placeholder:
        image = placeholder_image(ep, scene)
    duration = scene_duration(ep, scene)

    key = hashlib.sha1(json.dumps([file_hash(image), asdict(spec), duration], sort_keys=True,
                                  default=str).encode()).hexdigest()[:12]
    out_dir = ep.build_dir / "previews"
    out = out_dir / f"{scene.id}_{key}.mp4"
    if not out.exists():
        for old in out_dir.glob(f"{scene.id}_*.mp4"):
            old.unlink()
        r = SceneRenderer(image, spec, size=PREVIEW_SIZE, fps=24, duration=duration, flat=placeholder)
        r.render(out, crf=26, encoder_preset="ultrafast")
    return {"path": out, "duration": duration, "spec": asdict(spec), "placeholder": placeholder}
