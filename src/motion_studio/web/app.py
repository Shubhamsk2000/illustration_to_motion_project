"""Motion Studio web app: an editor for episodes (script, scenes, looks, previews, builds).

Run with `motion-studio` and open http://127.0.0.1:8010
"""

import hashlib
import os
import re
from dataclasses import asdict
from pathlib import Path

import cv2
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from motion_studio.episode import audio
from motion_studio.episode.build import scene_specs
from motion_studio.episode.preview import preview_scene, scene_duration
from motion_studio.episode.project import Episode, Settings, look_choices
from motion_studio.paths import EPISODES_DIR, WEB_STATIC_DIR
from motion_studio.scene import PRESETS
from motion_studio.web.jobs import JobManager

NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,60}$")
AUDIO_RE = re.compile(r"^(narration|music)\.(wav|mp3|m4a|flac|aac|ogg)$")
SERVABLE = ("images/", "export/", ".build/previews/", ".build/placeholders/", ".build/thumbs/")
MAX_UPLOAD = 300 * 1024 * 1024

app = FastAPI(title="Motion Studio")
jobs = JobManager()


def episodes_root() -> Path:
    """episodes/ in the project, or MOTION_STUDIO_EPISODES if set."""
    return Path(os.environ.get("MOTION_STUDIO_EPISODES") or EPISODES_DIR)


def episode(name: str, must_exist: bool = True) -> Episode:
    if not NAME_RE.match(name):
        raise HTTPException(400, "Episode names use lowercase letters, numbers, - and _")
    ep = Episode(episodes_root() / name)
    if must_exist and not ep.script_path.exists():
        raise HTTPException(404, f"No episode '{name}'")
    return ep


def file_url(ep: Episode, path: Path) -> str:
    rel = path.relative_to(ep.dir).as_posix()
    return f"/files/{ep.dir.name}/{rel}?v={int(path.stat().st_mtime)}"


def thumbnail(ep: Episode, image: Path) -> Path:
    key = hashlib.sha1(f"{image.name}:{image.stat().st_mtime_ns}:{image.stat().st_size}".encode()).hexdigest()[:10]
    out = ep.build_dir / "thumbs" / f"{image.stem}_{key}.jpg"
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        for old in out.parent.glob(f"{image.stem}_*.jpg"):
            old.unlink()
        img = cv2.imread(str(image))
        h, w = img.shape[:2]
        s = 360 / max(h, w)
        cv2.imwrite(str(out), cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA),
                    [cv2.IMWRITE_JPEG_QUALITY, 85])
    return out


def _errors(fn):
    """Turn validation problems into readable 400 responses."""
    try:
        return fn()
    except (ValueError, KeyError, FileNotFoundError, FileExistsError) as e:
        raise HTTPException(400, str(e).strip("'")) from e


# ---- episodes ----------------------------------------------------------------------------------------------

@app.get("/api/options")
def options():
    return {"looks": look_choices(), "presets": PRESETS, "settings": asdict(Settings())}


@app.get("/api/episodes")
def list_episodes():
    root = episodes_root()
    out = []
    for d in sorted(root.iterdir() if root.is_dir() else []):
        ep = Episode(d)
        if not (d.is_dir() and NAME_RE.match(d.name) and ep.script_path.exists()):
            continue
        script = ep.script()
        images = ep.images()
        exports = sorted(ep.export_dir.glob("*.mp4")) if ep.export_dir.is_dir() else []
        first = next((images[k] for k in sorted(images)), None)
        out.append({
            "name": d.name, "title": script.title, "scenes": len(script.scenes),
            "images": sum(1 for s in script.scenes if int(s.id) in images), "words": script.word_count,
            "narration": ep.narration() is not None, "exports": [p.name for p in exports],
            "thumb": file_url(ep, thumbnail(ep, first)) if first else None,
            "job": jobs.status(d.name),
        })
    return out


class NewEpisode(BaseModel):
    name: str
    title: str = ""


@app.post("/api/episodes")
def create_episode(body: NewEpisode):
    ep = episode(body.name, must_exist=False)
    if ep.dir.exists():
        raise HTTPException(400, f"'{body.name}' already exists")
    episodes_root().mkdir(parents=True, exist_ok=True)
    ep.create(body.title.strip() or body.name.replace("-", " ").title())
    ep.plan()
    return episode_state(body.name)


@app.get("/api/episodes/{name}")
def episode_state(name: str):
    ep = episode(name)
    report = _errors(ep.plan)  # refresh yaml + shot list from the script, keep edits
    settings = ep.settings()
    images = ep.images()
    scenes = []
    for sc in scene_specs(ep):
        s = sc["scene"]
        img = images.get(int(s.id))
        scenes.append({
            "id": s.id, "chapter": s.chapter, "text": s.text, "note": s.image_note, "words": len(s.words),
            "seconds": scene_duration(ep, s),
            "image": {"thumb": file_url(ep, thumbnail(ep, img)), "full": file_url(ep, img), "name": img.name}
            if img else None,
            "auto": {"look": sc["auto"], "word": sc["auto_word"]} if sc["auto"] else None,
            "override": sc["override"], "spec": asdict(sc["spec"]),
        })
    narration = ep.narration()
    music = ep.music(settings) if (settings.music is None or (ep.dir / settings.music).exists()) else None
    exports = []
    if ep.export_dir.is_dir():
        for p in sorted(ep.export_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
            if p.suffix in (".mp4", ".srt", ".txt"):
                exports.append({"name": p.name, "url": file_url(ep, p), "size": p.stat().st_size,
                                "mtime": int(p.stat().st_mtime)})
    chapters_txt = ep.export_dir / "chapters.txt"
    return {
        "name": name, "title": report["title"], "summary": report, "settings": asdict(settings),
        "script": ep.script_path.read_text(encoding="utf-8"), "scenes": scenes,
        "narration": {"name": narration.name, "seconds": round(audio.duration(narration), 1),
                      "url": file_url(ep, narration)} if narration else None,
        "music": {"name": music.name, "url": file_url(ep, music)} if music else None,
        "exports": exports, "chapters": chapters_txt.read_text() if chapters_txt.exists() else None,
        "job": jobs.status(name),
    }


class ScriptBody(BaseModel):
    text: str


@app.put("/api/episodes/{name}/script")
def save_script(name: str, body: ScriptBody):
    ep = episode(name)
    ep.script_path.write_text(body.text, encoding="utf-8")
    return episode_state(name)


@app.put("/api/episodes/{name}/settings")
def save_settings(name: str, body: dict):
    ep = episode(name)
    _errors(lambda: ep.update_settings(body))
    return episode_state(name)


@app.put("/api/episodes/{name}/scenes/{scene_id}")
def save_scene_look(name: str, scene_id: str, body: dict):
    ep = episode(name)
    _errors(lambda: ep.set_scene_look(scene_id, body))
    return episode_state(name)


async def _read_upload(file: UploadFile) -> tuple[bytes, str]:
    data = await file.read()
    if len(data) > MAX_UPLOAD:
        raise HTTPException(400, "File is too large")
    return data, Path(file.filename or "").suffix.lower()


@app.post("/api/episodes/{name}/scenes/{scene_id}/image")
async def upload_image(name: str, scene_id: str, file: UploadFile = File(...)):
    ep = episode(name)
    data, ext = await _read_upload(file)
    if not any(s.id == f"{int(scene_id):03d}" for s in ep.script().scenes):
        raise HTTPException(404, f"No scene {scene_id}")
    await run_in_threadpool(lambda: _errors(lambda: ep.save_image(scene_id, data, ext)))
    return episode_state(name)


@app.post("/api/episodes/{name}/audio/{kind}")
async def upload_audio(name: str, kind: str, file: UploadFile = File(...)):
    ep = episode(name)
    data, ext = await _read_upload(file)
    _errors(lambda: ep.save_audio(kind, data, ext))
    if kind == "music":
        _errors(lambda: ep.update_settings({"music": None}))  # use the uploaded music.* file
    return episode_state(name)


@app.post("/api/episodes/{name}/scenes/{scene_id}/preview")
async def scene_preview(name: str, scene_id: str):
    ep = episode(name)
    sid = f"{int(scene_id):03d}"
    result = await run_in_threadpool(lambda: _errors(lambda: preview_scene(ep, sid)))
    return {"url": file_url(ep, result["path"]), "seconds": result["duration"], "spec": result["spec"],
            "placeholder": result["placeholder"]}


# ---- builds -------------------------------------------------------------------------------------------------

class BuildBody(BaseModel):
    draft: bool = True


@app.post("/api/episodes/{name}/build")
def start_build(name: str, body: BuildBody):
    ep = episode(name)
    if ep.narration() is None:
        raise HTTPException(400, "Add your narration recording first.")
    try:
        jobs.start(name, ep.dir, body.draft)
    except RuntimeError as e:
        raise HTTPException(409, str(e)) from e
    return jobs.status(name)


@app.get("/api/episodes/{name}/build")
def build_status(name: str):
    episode(name)
    return jobs.status(name)


@app.delete("/api/episodes/{name}/build")
def cancel_build(name: str):
    episode(name)
    return {"cancelled": jobs.cancel(name)}


# ---- files + page -------------------------------------------------------------------------------------------

@app.get("/files/{name}/{path:path}")
def episode_file(name: str, path: str):
    ep = episode(name)
    if not (path.startswith(SERVABLE) or AUDIO_RE.match(path)):
        raise HTTPException(404)
    full = (ep.dir / path).resolve()
    if not full.is_relative_to(ep.dir.resolve()) or not full.is_file():
        raise HTTPException(404)
    return FileResponse(full)


app.mount("/static", StaticFiles(directory=WEB_STATIC_DIR), name="static")


@app.get("/")
def index():
    return FileResponse(WEB_STATIC_DIR / "index.html")


def main():
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="Motion Studio web app")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8010)  # 8000 is used by the Shorts Studio
    args = parser.parse_args()
    episodes_root().mkdir(parents=True, exist_ok=True)
    print(f"Motion Studio: http://{args.host}:{args.port}  (episodes in {episodes_root()})")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
