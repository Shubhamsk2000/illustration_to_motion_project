"""Every project path in one place, so no module hard-codes file locations."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

ASSETS = ROOT / "assets"
SAMPLES_DIR = ASSETS / "samples"        # reference illustrations (my favourite art style)
DEPTH_DIR = ASSETS / "depth"            # precomputed depth maps, <stem>.png
EXPERIMENTS_DIR = ROOT / "experiments"  # rendered comparisons of approaches 1-4
MODELS_DIR = ROOT / "models"            # downloaded model weights (git-ignored)
OUTPUT_DIR = ROOT / "output"            # everything you render (git-ignored)
EPISODES_DIR = ROOT / "episodes"        # your episodes (git-ignored)
DEPTH_CACHE_DIR = OUTPUT_DIR / "depth_cache"
CUSTOM_RENDERS_DIR = OUTPUT_DIR / "custom_renders"

WEB_STATIC_DIR = Path(__file__).resolve().parent / "web" / "static"

# The four reference scenes used across experiments and the web gallery.
SAMPLE_NAMES = ["warriors", "army", "king", "fox"]


def sample_image(name: str) -> Path:
    return SAMPLES_DIR / f"{name}.png"


def sample_depth(name: str) -> Path:
    return DEPTH_DIR / f"{name}.png"
