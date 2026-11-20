"""Time every script word against the narration recording.

Speech recognition (faster-whisper, CPU) gives timed words; those are matched to the script's words, and script
words that weren't recognised (names, numbers, stumbles) get times interpolated from their neighbours. The script
stays the source of truth for the text, so subtitles are spelled the way you wrote them.
"""

import hashlib
import json
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

from motion_studio.episode.script import Script, normalize_word


@dataclass
class TimedWord:
    text: str
    start: float
    end: float


def _load_16k(path):
    """Decode any audio file to 16 kHz mono float32 with FFmpeg (what whisper expects)."""
    import subprocess

    import numpy as np

    raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(path), "-f", "f32le", "-ac", "1", "-ar", "16000",
                          "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32)


def transcribe(audio_path, cache_dir, model: str = "base", language: str = "en") -> list[dict]:
    """[{word, start, end}] for the recording, cached by audio content + model."""
    data = Path(audio_path).read_bytes()
    key = hashlib.sha1(data + model.encode() + language.encode()).hexdigest()[:16]
    cache = Path(cache_dir) / f"transcript_{key}.json"
    if cache.exists():
        return json.loads(cache.read_text())

    from faster_whisper import WhisperModel

    print(f"[align] Transcribing narration with whisper '{model}' (first time only, about 1 minute per 8 minutes of audio)...")
    wm = WhisperModel(model, device="cpu", compute_type="int8")
    # No initial prompt: tested on a narration, a script prompt made it ~3x slower and garbled the opening.
    segments, info = wm.transcribe(_load_16k(audio_path), language=language, word_timestamps=True, vad_filter=True,
                                   beam_size=1)
    words = []
    for seg in segments:
        for w in seg.words or []:
            words.append({"word": w.word.strip(), "start": round(w.start, 3), "end": round(w.end, 3)})
        print(f"\r[align] {seg.end / max(info.duration, 1e-6) * 100:5.1f}%", end="", flush=True)
    print()
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(words))
    return words


def align_words(script_words: list[str], asr: list[dict], audio_seconds: float) -> tuple[list[TimedWord], float]:
    """Times for each script word, and the share of script words that were matched directly."""
    s_norm = [normalize_word(w) for w in script_words]
    a_norm = [normalize_word(w["word"]) for w in asr]
    n = len(script_words)
    starts: list[float | None] = [None] * n
    ends: list[float | None] = [None] * n

    matched = 0
    for blk in SequenceMatcher(None, s_norm, a_norm, autojunk=False).get_matching_blocks():
        for k in range(blk.size):
            i, j = blk.a + k, blk.b + k
            if s_norm[i]:
                starts[i], ends[i] = asr[j]["start"], asr[j]["end"]
                matched += 1

    # Interpolate unmatched words between the nearest matched neighbours.
    anchors = [-1] + [i for i in range(n) if starts[i] is not None] + [n]
    for i0, i1 in zip(anchors, anchors[1:]):
        missing = i1 - i0 - 1
        if missing <= 0:
            continue
        t_from = 0.0 if i0 < 0 else ends[i0]
        t_to = audio_seconds if i1 >= n else starts[i1]
        step = max(0.0, t_to - t_from) / missing
        for k in range(missing):
            starts[i0 + 1 + k] = t_from + step * k
            ends[i0 + 1 + k] = t_from + step * (k + 1)

    # Keep times monotonic.
    out, last = [], 0.0
    for w, s, e in zip(script_words, starts, ends):
        s = max(s, last)
        e = max(e, s + 0.05)
        out.append(TimedWord(w, round(s, 3), round(e, 3)))
        last = s
    return out, matched / max(1, sum(1 for w in s_norm if w))


@dataclass
class SceneTiming:
    id: str
    start: float  # when the image appears
    end: float    # when the next image takes over
    first_word: float
    last_word: float


def scene_timings(script: Script, words: list[TimedWord], total_seconds: float, lead: float = 0.25,
                  min_scene: float = 1.5) -> list[SceneTiming]:
    """Cut to each scene's image slightly before its first word; the last image holds until total_seconds."""
    out, idx = [], 0
    bounds = []
    for s in script.scenes:
        n = len(s.words)
        sw = words[idx:idx + n]
        idx += n
        bounds.append((s.id, sw[0].start, sw[-1].end))

    starts = []
    for k, (_, first, _) in enumerate(bounds):
        b = 0.0 if k == 0 else max(first - lead, starts[-1] + min_scene)
        starts.append(b)
    for k, (sid, first, last) in enumerate(bounds):
        end = starts[k + 1] if k + 1 < len(starts) else max(total_seconds, starts[k] + min_scene)
        out.append(SceneTiming(sid, round(starts[k], 3), round(end, 3), first, last))
    return out
