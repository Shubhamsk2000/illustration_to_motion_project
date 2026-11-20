"""Subtitles (.srt) from timed script words, and YouTube chapter timestamps."""

from motion_studio.episode.align import SceneTiming, TimedWord
from motion_studio.episode.script import Script

MAX_LINE = 42      # characters per subtitle line
MAX_LINES = 2
MAX_CUE_SECONDS = 6.0


def _ts(sec: float) -> str:
    ms = int(round(sec * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _wrap(words: list[str]) -> str:
    lines, cur = [], ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > MAX_LINE:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    lines.append(cur)
    return "\n".join(lines)


def build_cues(words: list[TimedWord]) -> list[tuple[float, float, str]]:
    cues, cur = [], []

    def fits(ws):
        text = _wrap([w.text for w in ws])
        return text.count("\n") < MAX_LINES and (ws[-1].end - ws[0].start) <= MAX_CUE_SECONDS

    for w in words:
        if cur and not fits(cur + [w]):
            cues.append(cur)
            cur = []
        cur.append(w)
        # Prefer to end a cue at a sentence end once it has some length.
        if w.text[-1:] in ".!?" and len(" ".join(x.text for x in cur)) > 20:
            cues.append(cur)
            cur = []
    if cur:
        cues.append(cur)

    out = []
    for k, ws in enumerate(cues):
        start = ws[0].start
        end = ws[-1].end + 0.25
        if k + 1 < len(cues):
            end = min(end, cues[k + 1][0].start - 0.02)
        out.append((start, max(end, start + 0.5), _wrap([w.text for w in ws])))
    return out


def write_srt(words: list[TimedWord], path) -> int:
    cues = build_cues(words)
    blocks = [f"{i}\n{_ts(s)} --> {_ts(e)}\n{text}\n" for i, (s, e, text) in enumerate(cues, 1)]
    path.write_text("\n".join(blocks), encoding="utf-8")
    return len(cues)


def _yt_time(sec: float) -> str:
    sec = int(sec)
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def chapters(script: Script, timings: list[SceneTiming], total: float) -> tuple[list[tuple[float, str]], list[str]]:
    """Chapter (start, title) list and any reasons YouTube would not show them."""
    starts = {t.id: t.start for t in timings}
    out = []
    for s in script.scenes:
        if s.chapter and (not out or out[-1][1] != s.chapter):
            out.append((starts[s.id], s.chapter))
    if out:
        out[0] = (0.0, out[0][1])  # YouTube requires the first chapter at 0:00
    problems = []
    if 0 < len(out) < 3:
        problems.append(f"only {len(out)} chapter(s); YouTube needs at least 3")
    for (a, title), (b, _) in zip(out, out[1:] + [(total, "")]):
        if b - a < 10:
            problems.append(f"chapter '{title}' is shorter than 10 seconds")
    return out, problems


def write_chapters(chs: list[tuple[float, str]], path) -> None:
    path.write_text("\n".join(f"{_yt_time(t)} {title}" for t, title in chs) + "\n", encoding="utf-8")
