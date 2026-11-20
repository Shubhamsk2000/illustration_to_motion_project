"""Episode builds run as separate processes, so the web server stays responsive and a build can be cancelled.

Only one build runs at a time: this machine has 2 CPU cores and a second build would just slow both down.
"""

import json
import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Job:
    episode: str
    draft: bool
    folder: Path
    proc: subprocess.Popen
    started: float = field(default_factory=time.time)
    finished: float | None = None
    cancelled: bool = False

    @property
    def job_dir(self) -> Path:
        return self.folder / ".build" / "job"

    def status(self) -> dict:
        code = self.proc.poll()
        if code is None:
            state = "running"
        elif self.cancelled:
            state = "cancelled"
        else:
            state = "done" if code == 0 else "failed"
        if code is not None and self.finished is None:
            self.finished = time.time()
        out = {"episode": self.episode, "draft": self.draft, "state": state,
               "elapsed": round((self.finished or time.time()) - self.started, 1),
               "stage": None, "progress": 0.0, "detail": ""}
        prog = _read_json(self.job_dir / "progress.json")
        if prog:
            out.update(stage=prog.get("stage"), progress=prog.get("progress", 0.0), detail=prog.get("detail", ""))
        if state == "done":
            out["progress"] = 1.0
            out["result"] = _read_json(self.job_dir / "progress.json.result.json")
        if state == "failed":
            out["error"] = _error_from_log(self.job_dir / "build.log")
        return out


def _read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _error_from_log(path: Path) -> str:
    try:
        lines = [ln for ln in path.read_text(errors="replace").replace("\r", "\n").splitlines() if ln.strip()]
    except OSError:
        return "The build stopped unexpectedly."
    for ln in reversed(lines):
        if ln.startswith("Error:"):
            return ln[len("Error:"):].strip()
    return lines[-1] if lines else "The build stopped unexpectedly."


class JobManager:
    def __init__(self, command=None):
        # command(folder, draft, progress_file) -> argv; replaceable in tests
        self.command = command or (lambda folder, draft, progress: [
            sys.executable, "-m", "motion_studio.episode.cli", "build", str(folder),
            "--draft" if draft else "--allow-missing", "--progress-file", str(progress)])
        self.jobs: dict[str, Job] = {}
        self.lock = threading.Lock()

    def running(self) -> Job | None:
        return next((j for j in self.jobs.values() if j.proc.poll() is None), None)

    def start(self, episode: str, folder: Path, draft: bool) -> Job:
        with self.lock:
            busy = self.running()
            if busy:
                raise RuntimeError(f"A build of '{busy.episode}' is already running. Wait for it or cancel it.")
            job_dir = folder / ".build" / "job"
            job_dir.mkdir(parents=True, exist_ok=True)
            for f in job_dir.iterdir():
                f.unlink()
            log = open(job_dir / "build.log", "w")
            proc = subprocess.Popen(self.command(folder, draft, job_dir / "progress.json"), stdout=log,
                                    stderr=subprocess.STDOUT, start_new_session=True,
                                    env={**os.environ, "PYTHONUNBUFFERED": "1"})
            log.close()
            job = Job(episode, draft, folder, proc)
            self.jobs[episode] = job
            return job

    def status(self, episode: str) -> dict | None:
        job = self.jobs.get(episode)
        return job.status() if job else None

    def cancel(self, episode: str) -> bool:
        job = self.jobs.get(episode)
        if not job or job.proc.poll() is not None:
            return False
        job.cancelled = True
        try:  # the build starts ffmpeg children: stop the whole process group
            os.killpg(job.proc.pid, signal.SIGTERM)
            job.proc.wait(timeout=10)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            os.killpg(job.proc.pid, signal.SIGKILL)
        return True
