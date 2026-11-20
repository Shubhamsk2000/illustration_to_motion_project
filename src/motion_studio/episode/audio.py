"""Voice clean-up and the final audio mix (FFmpeg filters only, runs on CPU)."""

import json
import re
import subprocess
from pathlib import Path

# Clean-up chain for a home narration recording:
#   rumble cut -> noise reduction -> less boxiness -> more clarity -> softer "s" sounds
#   -> even out loud/quiet words -> catch peaks
VOICE_CHAIN = ",".join([
    "highpass=f=70",
    # Non-local-means denoiser: on a test narration with room hiss it removed ~19 dB of noise vs ~4 dB for
    # afftdn, with no measurable loss of speech detail. Costs ~8 s per minute of audio (cached).
    "anlmdn=s=7",
    "equalizer=f=250:t=q:w=1.2:g=-2",
    "equalizer=f=3500:t=q:w=1.5:g=2.5",
    "deesser=i=0.4",
    "acompressor=threshold=0.1:ratio=3:attack=10:release=150:makeup=2",
    "alimiter=limit=0.95",
])


def run(cmd: list[str]) -> subprocess.CompletedProcess:
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd[:6])} ...\n{res.stderr[-2000:]}")
    return res


def duration(path) -> float:
    out = run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)])
    return float(out.stdout.strip())


def to_wav(src, dst, enhance: bool) -> None:
    """Narration -> 48 kHz mono WAV, cleaned up unless enhance is False."""
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    af = ["-af", VOICE_CHAIN] if enhance else []
    run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), *af, "-ar", "48000", "-ac", "1", str(dst)])


def mix(voice_wav, out_wav, total_seconds: float, music=None, music_volume: float = 0.15) -> None:
    """Voice padded to the video length, plus optional music that loops, ducks under the voice and fades out."""
    d = f"{total_seconds:.3f}"
    if music is None:
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(voice_wav),
               "-af", f"apad=whole_dur={d},atrim=0:{d},aformat=sample_rates=48000:channel_layouts=stereo",
               str(out_wav)]
    else:
        fade_out = max(0.0, total_seconds - 4.0)
        graph = (
            f"[0:a]apad=whole_dur={d},atrim=0:{d},aformat=sample_rates=48000:channel_layouts=stereo,asplit[v][sc];"
            f"[1:a]aformat=sample_rates=48000:channel_layouts=stereo,atrim=0:{d},volume={music_volume},"
            f"afade=t=in:d=2,afade=t=out:st={fade_out:.3f}:d=4[m];"
            "[m][sc]sidechaincompress=threshold=0.03:ratio=6:attack=50:release=600[md];"
            "[v][md]amix=inputs=2:normalize=0:duration=first[out]"
        )
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(voice_wav), "-stream_loop", "-1", "-i", str(music),
               "-filter_complex", graph, "-map", "[out]", str(out_wav)]
    run(cmd)


def measure_loudness(path, target: float) -> dict:
    res = run(["ffmpeg", "-hide_banner", "-i", str(path), "-af",
               f"loudnorm=I={target}:TP=-1.5:LRA=11:print_format=json", "-f", "null", "-"])
    block = re.findall(r"\{[^{}]*\}", res.stderr, flags=re.S)[-1]
    return json.loads(block)


def mux_final(video, audio_wav, out, target: float = -14.0) -> dict:
    """Two-pass loudness normalisation of the mix, muxed with the (already encoded) video."""
    m = measure_loudness(audio_wav, target)
    ln = (f"loudnorm=I={target}:TP=-1.5:LRA=11:measured_I={m['input_i']}:measured_TP={m['input_tp']}:"
          f"measured_LRA={m['input_lra']}:measured_thresh={m['input_thresh']}:offset={m['target_offset']}:"
          "linear=true,aresample=48000")
    run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(video), "-i", str(audio_wav), "-map", "0:v", "-map", "1:a",
         "-c:v", "copy", "-af", ln, "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(out)])
    return m
