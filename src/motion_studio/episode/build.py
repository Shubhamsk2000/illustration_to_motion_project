"""Build a finished episode: clean voice -> time scenes -> render -> mix -> export."""

import hashlib
import json
import re
import time
from dataclasses import asdict, replace
from pathlib import Path

from motion_studio.camera import auto_move
from motion_studio.episode import audio
from motion_studio.episode.align import align_words, scene_timings, transcribe
from motion_studio.episode.captions import chapters, write_chapters, write_srt
from motion_studio.episode.project import Episode
from motion_studio.episode.timeline import Shot, make_placeholder, render_timeline
from motion_studio.scene import PRESETS, spec_from_preset

DRAFT_SIZE = (960, 540)


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:60] or "episode"


def clean_voice(ep: Episode, enhance: bool) -> Path:
    src = ep.narration()
    if src is None:
        raise FileNotFoundError(f"No narration found. Save your recording as {ep.dir}/narration.wav (or .mp3/.m4a)")
    key = hashlib.sha1(src.read_bytes() + audio.VOICE_CHAIN.encode() + bytes([enhance])).hexdigest()[:12]
    out = ep.build_dir / f"voice_{key}.wav"
    if not out.exists():
        print("[voice] Cleaning up narration..." if enhance else "[voice] Converting narration...")
        for old in ep.build_dir.glob("voice_*.wav"):
            old.unlink()
        audio.to_wav(src, out, enhance)
    return out


def scene_spec(index: int, default_preset: str, override: dict):
    """Per-scene look: the scene's own preset/settings, else the default preset with a varied camera move."""
    preset = override.get("preset", default_preset)
    spec = spec_from_preset(preset, **{k: v for k, v in override.items() if k != "preset"})
    if "move" not in override and "preset" not in override:
        spec = replace(spec, move=auto_move(index))
    return replace(spec, loop=False)


def build(folder, draft: bool = False, allow_missing: bool = False) -> dict:
    t0 = time.time()
    ep = Episode(folder)
    script, settings, overrides = ep.script(), ep.settings(), ep.overrides()
    if not script.scenes:
        raise ValueError("script.md has no scenes (paragraphs).")
    if settings.default_preset not in PRESETS:
        raise ValueError(f"default_preset '{settings.default_preset}' is not one of: {', '.join(PRESETS)}")
    ep.build_dir.mkdir(parents=True, exist_ok=True)

    images = ep.images()
    missing = [s.id for s in script.scenes if int(s.id) not in images]
    if missing and not (draft or allow_missing):
        raise FileNotFoundError(
            f"{len(missing)} image(s) missing: {', '.join(missing[:12])}{' ...' if len(missing) > 12 else ''}. "
            "Add them to images/, or use --draft / --allow-missing to render placeholder cards.")

    # 1. Voice and timing
    voice = clean_voice(ep, settings.enhance_voice)
    voice_seconds = audio.duration(voice)
    all_words = [w for s in script.scenes for w in s.words]
    asr = transcribe(voice, ep.build_dir, settings.whisper_model, settings.language)
    words, match = align_words(all_words, asr, voice_seconds)
    total = voice_seconds + settings.end_hold
    timings = scene_timings(script, words, total)

    # 2. Shots
    size = DRAFT_SIZE if draft else tuple(int(v) for v in settings.size.lower().split("x"))
    shots = []
    for i, (s, tm) in enumerate(zip(script.scenes, timings)):
        img = images.get(int(s.id))
        placeholder = img is None
        if placeholder:
            img = ep.build_dir / "placeholders" / f"{s.id}.png"
            if not img.exists():
                make_placeholder(img, s.id, s.text)
        shots.append(Shot(s.id, img, scene_spec(i, settings.default_preset, overrides.get(s.id, {})),
                          tm.start, tm.end, placeholder))

    # 3. Video
    suffix = "_draft" if draft else ""
    video = ep.build_dir / f"video{suffix}.mp4"
    stats = render_timeline(shots, video, ep.build_dir, size=size, fps=settings.fps,
                            transition=settings.transition, draft=draft)

    # 4. Audio mix + final file
    music = ep.music(settings)
    mix_wav = ep.build_dir / "mix.wav"
    audio.mix(voice, mix_wav, total, music, settings.music_volume)
    ep.export_dir.mkdir(parents=True, exist_ok=True)
    final = ep.export_dir / f"{slugify(script.title)}{suffix}.mp4"
    loud = audio.mux_final(video, mix_wav, final, settings.loudness)

    # 5. Subtitles, chapters, timeline
    n_cues = write_srt(words, ep.export_dir / "subtitles.srt")
    chs, chapter_problems = chapters(script, timings, total)
    if chs:
        write_chapters(chs, ep.export_dir / "chapters.txt")
    long_scenes = [t.id for t in timings if t.end - t.start > 16]
    (ep.export_dir / "timeline.json").write_text(json.dumps([
        {"id": sh.id, "start": sh.start, "end": sh.end, "image": str(sh.image), "placeholder": sh.placeholder,
         "spec": asdict(sh.spec)} for sh in shots], indent=2))

    return {
        "final": str(final), "seconds": round(total, 1), "scenes": len(shots), "missing_images": missing,
        "script_match": round(match, 3), "subtitle_cues": n_cues, "chapters": len(chs),
        "chapter_problems": chapter_problems, "long_scenes": long_scenes, "music": str(music) if music else None,
        "input_loudness": loud.get("input_i"), "segments_rendered": stats["rendered"],
        "segments_reused": stats["reused"], "build_seconds": round(time.time() - t0, 1),
    }
