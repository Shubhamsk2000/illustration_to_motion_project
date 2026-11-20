"""Stream raw frames into FFmpeg (H.264 MP4)."""

import subprocess
from pathlib import Path


class VideoWriter:
    def __init__(self, path, width: int, height: int, fps: float, crf: int = 18, preset: str = "veryfast"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.path = str(path)
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
            "-c:v", "libx264", "-preset", preset, "-crf", str(crf), "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", self.path,
        ]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    def write(self, frame) -> None:
        self.proc.stdin.write(frame.tobytes())

    def close(self) -> None:
        self.proc.stdin.close()
        err = self.proc.stderr.read().decode(errors="replace")
        if self.proc.wait() != 0:
            raise RuntimeError(f"FFmpeg failed writing {self.path}:\n{err}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, *_):
        if exc_type is None:
            self.close()
        else:
            self.proc.kill()
            self.proc.wait()


def make_preview_gif(mp4_path, gif_path, width: int = 480, fps: int = 12) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp4_path), "-vf",
         f"fps={fps},scale={width}:-1:flags=lanczos,split[a][b];[a]palettegen[p];[b][p]paletteuse", str(gif_path)],
        check=True,
    )
