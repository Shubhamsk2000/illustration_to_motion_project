"""An episode folder and its files.

episodes/roman-soldier/
  script.md            you write this (see script.SCRIPT_TEMPLATE)
  episode.yaml         settings + per-scene look overrides (created by `plan`, edit freely)
  shotlist.md          which image to make for each scene (created by `plan`)
  narration.wav        your recording (.wav/.mp3/.m4a/.flac)
  music.mp3            optional background music
  images/001.png ...   one image per scene (scene_1.png, 1.jpg ... also work)
  .build/              cache (cleaned voice, transcript, rendered segments)
  export/              final video, subtitles, chapters
"""

import re
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import yaml

from motion_studio.episode.looks import suggest_look
from motion_studio.episode.script import SCRIPT_TEMPLATE, WORDS_PER_MINUTE, Script, parse_script

AUDIO_EXTS = (".wav", ".mp3", ".m4a", ".flac", ".aac", ".ogg")
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")
SCENE_OVERRIDE_KEYS = ("preset", "move", "strength", "depth", "breathing", "atmosphere", "mist", "glow", "light",
                       "fit")


@dataclass
class Settings:
    fps: int = 24
    size: str = "1920x1080"
    transition: float = 0.6         # crossfade between scenes, seconds
    default_preset: str = "documentary"
    auto_look: bool = True          # pick each scene's look from its words (fire -> campfire, fog -> misty ...)
    enhance_voice: bool = True      # set false if you already cleaned the voice in another tool
    music: str | None = None        # file in the folder; music.* is picked up automatically
    music_volume: float = 0.15      # music level under the voice (it also ducks while you speak)
    loudness: float = -14.0         # final loudness, LUFS (YouTube plays at -14)
    whisper_model: str = "base"     # speech recognition used to time scenes ("small" is slower, rarely needed)
    language: str = "en"
    image_style: str = ""           # added to every prompt in shotlist.md, e.g. "flat 2D storybook illustration"
    end_hold: float = 1.5           # seconds the last image stays after the narration ends


class Episode:
    def __init__(self, folder):
        self.dir = Path(folder)
        self.script_path = self.dir / "script.md"
        self.yaml_path = self.dir / "episode.yaml"
        self.images_dir = self.dir / "images"
        self.build_dir = self.dir / ".build"
        self.export_dir = self.dir / "export"

    # ---- files -------------------------------------------------------------------------------------------

    def script(self) -> Script:
        if not self.script_path.exists():
            raise FileNotFoundError(f"No script.md in {self.dir}. Create one with: motion-episode new {self.dir}")
        return parse_script(self.script_path.read_text(encoding="utf-8"))

    def _yaml(self) -> dict:
        if self.yaml_path.exists():
            return yaml.safe_load(self.yaml_path.read_text(encoding="utf-8")) or {}
        return {}

    def settings(self) -> Settings:
        raw = self._yaml().get("settings") or {}
        known = {f.name for f in fields(Settings)}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"Unknown setting(s) in episode.yaml: {', '.join(sorted(unknown))}")
        return Settings(**raw)

    def overrides(self) -> dict[str, dict]:
        out = {}
        for item in self._yaml().get("scenes") or []:
            sid = f"{int(item['id']):03d}"
            chosen = {k: item[k] for k in SCENE_OVERRIDE_KEYS if item.get(k) is not None}
            if chosen:
                out[sid] = chosen
        return out

    def narration(self) -> Path | None:
        return self._find_audio("narration")

    def music(self, settings: Settings) -> Path | None:
        if settings.music:
            p = self.dir / settings.music
            if not p.exists():
                raise FileNotFoundError(f"Music file not found: {p}")
            return p
        return self._find_audio("music")

    def _find_audio(self, stem: str) -> Path | None:
        for ext in AUDIO_EXTS:
            p = self.dir / f"{stem}{ext}"
            if p.exists():
                return p
        return None

    def images(self) -> dict[int, Path]:
        """Scene number -> image file, matched by the number in the file name (001.png, scene_1.jpg ...)."""
        found: dict[int, Path] = {}
        if not self.images_dir.is_dir():
            return found
        for p in sorted(self.images_dir.iterdir()):
            m = re.search(r"(\d+)", p.stem)
            if p.suffix.lower() in IMAGE_EXTS and m:
                n = int(m.group(1))
                if n in found:
                    raise ValueError(f"Two images for scene {n}: {found[n].name} and {p.name}")
                found[n] = p
        return found

    # ---- edits (used by the web studio) -------------------------------------------------------------------

    def set_scene_look(self, scene_id: str, values: dict) -> dict:
        """Replace one scene's look overrides; empty/None values mean 'use the automatic choice'."""
        script = self.script()
        sid = f"{int(scene_id):03d}"
        if sid not in {s.id for s in script.scenes}:
            raise KeyError(f"No scene {scene_id}")
        clean = validate_look({k: v for k, v in values.items() if v not in (None, "")})
        overrides = self.overrides()
        if clean:
            overrides[sid] = clean
        else:
            overrides.pop(sid, None)
        self.write_yaml(self.settings(), overrides, script)
        return clean

    def update_settings(self, values: dict) -> Settings:
        current = asdict(self.settings())
        for k, v in values.items():
            if k not in current:
                raise ValueError(f"Unknown setting: {k}")
            default = getattr(Settings, k, None)
            if v is None or v == "":
                current[k] = None if k == "music" else default
            elif isinstance(default, bool):
                current[k] = v if isinstance(v, bool) else str(v).lower() in ("1", "true", "yes", "on")
            elif isinstance(default, int):
                current[k] = int(v)
            elif isinstance(default, float):
                current[k] = float(v)
            else:
                current[k] = str(v)
        settings = Settings(**current)
        from motion_studio.scene import PRESETS

        if settings.default_preset not in PRESETS:
            raise ValueError(f"Unknown preset: {settings.default_preset}")
        if not re.fullmatch(r"\d{2,4}x\d{2,4}", settings.size):
            raise ValueError("size must look like 1920x1080")
        script = self.script() if self.script_path.exists() else None
        self.write_yaml(settings, self.overrides(), script)
        if script:
            self.write_shotlist(script, settings)
        return settings

    def save_image(self, scene_id: str, data: bytes, ext: str) -> Path:
        """Store the image for a scene as images/<id><ext>, replacing any earlier image for that scene."""
        import cv2
        import numpy as np

        ext = ext.lower()
        if ext not in IMAGE_EXTS:
            raise ValueError(f"Images must be one of {', '.join(IMAGE_EXTS)}")
        if cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR) is None:
            raise ValueError("That file is not a readable image")
        n = int(scene_id)
        self.images_dir.mkdir(parents=True, exist_ok=True)
        old = self.images().get(n)
        if old is not None:
            old.unlink()
        path = self.images_dir / f"{n:03d}{ext}"
        path.write_bytes(data)
        return path

    def save_audio(self, kind: str, data: bytes, ext: str) -> Path:
        """Store narration.<ext> or music.<ext>, replacing the previous file of that kind."""
        if kind not in ("narration", "music"):
            raise ValueError("kind must be narration or music")
        ext = ext.lower()
        if ext not in AUDIO_EXTS:
            raise ValueError(f"Audio must be one of {', '.join(AUDIO_EXTS)}")
        for e in AUDIO_EXTS:
            (self.dir / f"{kind}{e}").unlink(missing_ok=True)
        path = self.dir / f"{kind}{ext}"
        path.write_bytes(data)
        return path

    # ---- commands ----------------------------------------------------------------------------------------

    def create(self, title: str) -> None:
        if self.script_path.exists():
            raise FileExistsError(f"{self.script_path} already exists")
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.script_path.write_text(SCRIPT_TEMPLATE.format(title=title), encoding="utf-8")
        self.write_yaml(Settings(), {})

    def write_yaml(self, settings: Settings, overrides: dict[str, dict], script: Script | None = None) -> None:
        scenes = []
        auto_on = settings.auto_look
        for s in (script.scenes if script else []):
            entry = {"id": s.id, "says": _short(s.text)}
            look, word = suggest_look(s.text, s.image_note) if auto_on else (None, None)
            if look:
                entry["auto_look"] = f"{look} (from '{word}')"
            entry.update(overrides.get(s.id, {}))
            scenes.append(entry)
        header = (
            "# Episode settings and per-scene looks. Edit freely; `motion-episode plan` keeps your changes.\n"
            "# Per scene you can set: preset, move, strength, depth, breathing, atmosphere, light, fit\n"
            "#   presets: documentary portrait epic calm campfire winter storm magic_forest misty sunlit battle\n"
            "#   moves: push_in pull_out pan_left pan_right tilt_up tilt_down static\n"
            "#   atmosphere: none dust embers fireflies snow rain\n"
            "#   mist: none fog smoke      glow: none rays firelight\n"
            "# auto_look is only information: what the scene's words suggest. Set preset: to choose yourself.\n"
        )
        body = yaml.safe_dump({"settings": asdict(settings), "scenes": scenes}, sort_keys=False,
                              allow_unicode=True, width=120)
        self.yaml_path.write_text(header + body, encoding="utf-8")

    def plan(self) -> dict:
        """Refresh episode.yaml and shotlist.md from script.md, keeping settings and per-scene overrides."""
        script = self.script()
        settings = self.settings() if self.yaml_path.exists() else Settings()
        old = self.overrides()
        self.write_yaml(settings, old, script)
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.write_shotlist(script, settings)

        images = self.images()
        long = [s.id for s in script.scenes if len(s.words) > 40]
        short = [s.id for s in script.scenes if len(s.words) < 5]
        minutes = script.word_count / WORDS_PER_MINUTE
        return {
            "title": script.title, "scenes": len(script.scenes), "words": script.word_count,
            "est_minutes": round(minutes, 1),
            "avg_scene_seconds": round(minutes * 60 / max(1, len(script.scenes)), 1),
            "chapters": script.chapters,
            "images_found": sum(1 for i in range(1, len(script.scenes) + 1) if i in images),
            "missing_images": [s.id for s in script.scenes if int(s.id) not in images],
            "long_scenes": long, "short_scenes": short,
            "dropped_overrides": sorted(set(old) - {s.id for s in script.scenes}),
            "looks": self._look_summary(script, settings, old),
        }

    def _look_summary(self, script: Script, settings: Settings, overrides: dict) -> dict[str, int]:
        counts: dict[str, int] = {}
        for s in script.scenes:
            auto = suggest_look(s.text, s.image_note)[0] if settings.auto_look else None
            look = overrides.get(s.id, {}).get("preset") or auto or settings.default_preset
            counts[look] = counts.get(look, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: -kv[1]))

    def write_shotlist(self, script: Script, settings: Settings) -> None:
        style = settings.image_style.strip()
        lines = [f"# Shot list: {script.title}", "",
                 f"{len(script.scenes)} images. Save each one as `images/<number>.png` at 16:9, "
                 "ideally 1920x1080 or larger.", ""]
        chapter = None
        for s in script.scenes:
            if s.chapter != chapter and s.chapter:
                chapter = s.chapter
                lines += [f"## {chapter}", ""]
            prompt = s.image_note or s.text
            if style:
                prompt = f"{prompt}. {style}"
            lines += [f"### {s.id}  →  images/{s.id}.png", "", f"Narration: {s.text}", "",
                      f"Prompt: {prompt}, 16:9", ""]
        (self.dir / "shotlist.md").write_text("\n".join(lines), encoding="utf-8")


def look_choices() -> dict[str, list]:
    from motion_studio.atmosphere import KIND_NAMES
    from motion_studio.camera import ALL_MOVES
    from motion_studio.effects import GLOW_NAMES, MIST_NAMES
    from motion_studio.scene import FITS, PRESETS

    return {"preset": list(PRESETS), "move": ALL_MOVES, "atmosphere": KIND_NAMES, "mist": MIST_NAMES,
            "glow": GLOW_NAMES, "fit": FITS}


NUMERIC_LOOK_LIMITS = {"strength": (0.0, 3.0), "depth": (0.0, 3.0), "breathing": (0.0, 0.03), "light": (0.0, 0.3)}


def validate_look(values: dict) -> dict:
    choices = look_choices()
    out = {}
    for k, v in values.items():
        if k in choices:
            if v not in choices[k]:
                raise ValueError(f"{k} must be one of: {', '.join(choices[k])}")
            out[k] = v
        elif k in NUMERIC_LOOK_LIMITS:
            lo, hi = NUMERIC_LOOK_LIMITS[k]
            v = float(v)
            if not lo <= v <= hi:
                raise ValueError(f"{k} must be between {lo} and {hi}")
            out[k] = v
        else:
            raise ValueError(f"Unknown look setting: {k}")
    return out


def _short(text: str, n: int = 70) -> str:
    return text if len(text) <= n else text[: n - 1].rsplit(" ", 1)[0] + "…"
