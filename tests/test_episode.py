import subprocess

import cv2
import numpy as np
import pytest

from conftest import synthetic_scene
from motion_studio.episode import audio
from motion_studio.episode.align import TimedWord, align_words, scene_timings
from motion_studio.episode.build import scene_spec
from motion_studio.episode.captions import build_cues, chapters
from motion_studio.episode.project import Episode
from motion_studio.episode.script import parse_script
from motion_studio.episode.timeline import Shot, render_timeline
from motion_studio.scene import SceneSpec

SCRIPT = """# The Test Episode
<!-- a comment -->
## Morning

First scene starts here. It has two sentences.
> Image: a tent at dawn

Second scene, a single line
that wraps onto two lines.

## Noon

Third scene.
> Image: a road
> with hills

## Night

Fourth scene by the fire.
"""


def test_parse_script():
    s = parse_script(SCRIPT)
    assert s.title == "The Test Episode"
    assert [x.id for x in s.scenes] == ["001", "002", "003", "004"]
    assert s.scenes[0].image_note == "a tent at dawn"
    assert s.scenes[1].text == "Second scene, a single line that wraps onto two lines."
    assert s.scenes[2].image_note == "a road with hills"
    assert s.chapters == ["Morning", "Noon", "Night"]
    assert s.scenes[3].chapter == "Night"
    assert "comment" not in " ".join(x.text for x in s.scenes)


def fake_asr(words, gap=0.4, drop=(), replace=None):
    """Timed 'recognised' words: one word per 0.4 s, optionally missing or misheard some."""
    out, t = [], 0.5
    for i, w in enumerate(words):
        if i not in drop:
            out.append({"word": (replace or {}).get(i, w), "start": t, "end": t + 0.3})
        t += gap
    return out


def test_alignment_interpolates_missed_words():
    words = "Marcus marched with the Twentieth Legion in 43 AD across Britain".split()
    asr = fake_asr(words, drop={5}, replace={7: "forty-three"})
    timed, match = align_words(words, asr, audio_seconds=10)
    assert match == pytest.approx(9 / 11)
    starts = [w.start for w in timed]
    assert starts == sorted(starts)
    assert timed[5].start == pytest.approx(2.5, abs=0.25)   # dropped word sits between its neighbours
    assert timed[7].start == pytest.approx(3.3, abs=0.25)   # misheard number too
    assert timed[0].start == pytest.approx(0.5)


def test_scene_timings_cut_before_first_word():
    script = parse_script(SCRIPT)
    words = [w for s in script.scenes for w in s.words]
    timed, _ = align_words(words, fake_asr(words), audio_seconds=12)
    tm = scene_timings(script, timed, total_seconds=13)
    assert tm[0].start == 0.0 and tm[-1].end == 13
    for a, b in zip(tm, tm[1:]):
        assert a.end == b.start
        # Cut just before the first word, unless the previous image would be on screen under 1.5 s.
        assert b.start == pytest.approx(max(b.first_word - 0.25, a.start + 1.5))
    assert any(b.start > b.first_word - 0.25 for b in tm[1:])  # the 2-word scene triggers the minimum


def test_subtitle_cues():
    words = [TimedWord(w, i * 0.4, i * 0.4 + 0.3) for i, w in enumerate(
        ("This is a long sentence that should be split into readable subtitle cues for viewers. "
         "Short one. Another quite long sentence follows here to fill the screen.").split())]
    cues = build_cues(words)
    for start, end, text in cues:
        assert end > start
        assert text.count("\n") <= 1
        assert all(len(line) <= 42 for line in text.split("\n"))
    assert all(a[1] <= b[0] for a, b in zip(cues, cues[1:]))


def test_chapters_follow_youtube_rules():
    script = parse_script(SCRIPT)
    words = [w for s in script.scenes for w in s.words]
    timed, _ = align_words(words, fake_asr(words, gap=1.0), audio_seconds=30)
    tm = scene_timings(script, timed, total_seconds=31)
    chs, problems = chapters(script, tm, 31)
    assert [c[1] for c in chs] == ["Morning", "Noon", "Night"]
    assert chs[0][0] == 0.0
    assert any("shorter than 10 seconds" in p for p in problems)  # Night is only a few seconds


def test_scene_spec_varies_moves_but_respects_overrides():
    a, b = scene_spec(0, "documentary", {}), scene_spec(1, "documentary", {})
    assert a.move != b.move and a.atmosphere == "dust"
    c = scene_spec(2, "documentary", {"preset": "campfire"})
    assert c.atmosphere == "embers" and c.move == "push_in"
    d = scene_spec(3, "documentary", {"move": "static", "strength": 0.5})
    assert d.move == "static" and d.strength == 0.5 and not d.loop


def test_plan_keeps_overrides(tmp_path):
    ep = Episode(tmp_path / "ep")
    ep.create("Test")
    ep.script_path.write_text(SCRIPT)
    ep.plan()
    text = ep.yaml_path.read_text().replace("- id: '002'", "- id: '002'\n  preset: campfire")
    ep.yaml_path.write_text(text.replace("transition: 0.6", "transition: 1.0"))
    (ep.images_dir / "scene_2.jpg").write_bytes(b"x")
    report = ep.plan()
    assert ep.overrides() == {"002": {"preset": "campfire"}}
    assert ep.settings().transition == 1.0
    assert report["images_found"] == 1 and report["missing_images"] == ["001", "003", "004"]
    assert (ep.dir / "shotlist.md").read_text().count("### ") == 4


def _frames(path):
    cap, out = cv2.VideoCapture(str(path)), []
    while True:
        ok, f = cap.read()
        if not ok:
            return out
        out.append(f.astype(np.int16))


def test_timeline_joins_smoothly_and_caches(tmp_path):
    shots = []
    for i, (start, end) in enumerate([(0.0, 1.5), (1.5, 2.75), (2.75, 4.0)]):
        img, _ = synthetic_scene(tmp_path, 640, 360, name=f"{i}.png")
        if i == 1:  # make scene 2 clearly different
            cv2.imwrite(str(img), 255 - cv2.imread(str(img)))
        shots.append(Shot(f"{i + 1:03d}", img, SceneSpec(move="push_in"), start, end, placeholder=True))
    out = tmp_path / "video.mp4"
    stats = render_timeline(shots, out, tmp_path / "build", size=(320, 180), fps=12, transition=0.5)
    assert stats == {"rendered": 3, "reused": 0}
    frames = _frames(out)
    assert len(frames) == 48
    diffs = [np.abs(b - a).mean() for a, b in zip(frames, frames[1:])]
    hard_cut = np.abs(frames[17] - frames[18]).mean()
    assert max(diffs) < 40, "a crossfade should never jump like a hard cut"
    assert hard_cut < 40

    # Changing only the last shot re-renders it and its neighbour, not the first one.
    shots[2].spec = SceneSpec(move="pull_out")
    stats = render_timeline(shots, out, tmp_path / "build", size=(320, 180), fps=12, transition=0.5)
    assert stats == {"rendered": 2, "reused": 1}


def test_mix_and_loudness(tmp_path):
    voice, music = tmp_path / "voice.wav", tmp_path / "music.wav"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "sine=f=300:d=3", "-ar", "48000", str(voice)], check=True)
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "sine=f=100:d=2", str(music)], check=True)
    mix = tmp_path / "mix.wav"
    audio.mix(voice, mix, 5.0, music, 0.2)
    assert audio.duration(mix) == pytest.approx(5.0, abs=0.05)
    video = tmp_path / "v.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=c=black:s=64x36:d=5:r=12", "-c:v",
                    "libx264", "-pix_fmt", "yuv420p", str(video)], check=True)
    final = tmp_path / "final.mp4"
    audio.mux_final(video, mix, final, -14.0)
    measured = audio.measure_loudness(final, -14.0)
    assert float(measured["input_i"]) == pytest.approx(-14.0, abs=1.0)
