# Motion Studio

Turns still AI illustrations into living video clips: subtle camera moves, depth parallax, breathing characters,
wind, light and particles. Built for a narrated YouTube channel of history stories ("a day in the life of…"),
where every narrated beat is one illustration brought to life.

Runs locally on CPU (no GPU needed). Depth comes from Depth-Anything-V2-Small, video encoding from FFmpeg.

## Setup

```bash
sudo apt install -y ffmpeg
python3 -m venv .venv
.venv/bin/pip install -e .
```

## Use

```bash
# Every image in a folder -> one 1920x1080 clip each (camera move varies scene to scene)
.venv/bin/motion-batch --input_dir my_scenes --output_dir output/my_episode --duration 7

# Quick 960x540 drafts to pick moves before the real render
.venv/bin/motion-batch --input_dir my_scenes --draft

# One image with a look, overriding parts of it
.venv/bin/motion-batch --image my_scenes/camp.png --preset campfire --move pull_out --duration 8

# Seamless loop
.venv/bin/motion-batch --image my_scenes/forest.png --preset magic_forest --loop --duration 6

# Review gallery + live parallax playground: http://127.0.0.1:8010
.venv/bin/motion-studio
```

- **Moves:** `push_in`, `pull_out`, `pan_left`, `pan_right`, `tilt_up`, `tilt_down`, `static`; loops: `drift`, `breathe`.
- **Atmosphere:** `dust`, `embers`, `fireflies`, `snow`, `rain`.
- **Presets:** `documentary`, `portrait`, `epic`, `calm`, `campfire`, `winter`, `storm`, `magic_forest`.
- **Fit:** 16:9 (or close) images fill the frame; portrait images are shown whole on a blurred backdrop
  (`--fit crop|blur` to force one). Generate images at 16:9, 1920 px wide or more, for the best result.
- Tuning flags: `--strength`, `--depth`, `--breathing`, `--light`, `--fps`, `--size`. See `motion-batch -h`.
- Depth maps and layer splits are cached in `output/`, so re-rendering an image is fast.

The old experimental approaches are still available with `--approach 1..4`.

## Tests

```bash
uv pip install --python .venv/bin/python -e ".[dev]"   # or: .venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest
```

`test_golden_frame` compares against a stored frame; after an intended change to the look, run it with
`UPDATE_GOLDEN=1` to accept the new frame.

## Layout

```
src/motion_studio/
  scene.py           scene engine: image -> 16:9 clip (framing, layered parallax, breathing, light)
  camera.py          documentary camera moves and loops
  layers.py          splits an image into background / subject / foreground planes
  atmosphere.py      particles: dust, embers, fireflies, snow, rain
  depth.py           Depth-Anything-V2 depth with caching
  encode.py          FFmpeg writer
  engine.py          older warp-based preset engine (kept for reference)
  approaches/        the four original experiments (a1..a4)
  batch.py           motion-batch CLI
  web_layers.py      exports bg/mid/fg layers for the web playground
  web/               FastAPI app (motion-studio) + static UI
  paths.py           every project path in one place
assets/
  samples/           reference illustrations in the target art style
  depth/             precomputed depth maps for the samples
experiments/         rendered comparisons of the approaches (videos are git-ignored)
docs/PLAN.md         improvement roadmap
tests/               pytest suite (+ golden frame)
output/              your renders (git-ignored)
models/, scratch/    model weights and old debug images (git-ignored)
```

Rendered videos are not in git. To rebuild an approach's comparison videos, run e.g.
`.venv/bin/python -m motion_studio.approaches.a3_multilayer_parallax`.
