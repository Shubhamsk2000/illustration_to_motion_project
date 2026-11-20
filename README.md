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
# Animate every image in a folder (default: assets/samples -> output/videos)
.venv/bin/motion-batch --approach 4 --input_dir my_scenes --output_dir output/my_episode --duration 6

# One image
.venv/bin/motion-batch --approach 3 --image my_scenes/market.png --output output/market.mp4

# Review gallery + live parallax playground: http://127.0.0.1:8010
.venv/bin/motion-studio
```

Approaches: `1` semantic shaders, `2` depth-mesh warp, `3` multi-layer 2D parallax, `4` layered parallax with
character breathing. The preset engine (`motion_studio.engine.render_animation`) adds camera moves, wind, sun glow
and particles on top of depth parallax.

## Layout

```
src/motion_studio/
  engine.py          preset engine: depth parallax + wind + light + particles
  approaches/        the four motion approaches (a1..a4)
  batch.py           motion-batch CLI
  web_layers.py      exports bg/mid/fg layers for the web playground
  web/               FastAPI app (motion-studio) + static UI
  paths.py           every project path in one place
assets/
  samples/           reference illustrations in the target art style
  depth/             precomputed depth maps for the samples
experiments/         rendered comparisons of the approaches (videos are git-ignored)
docs/PLAN.md         improvement roadmap
output/              your renders (git-ignored)
models/, scratch/    model weights and old debug images (git-ignored)
```

Rendered videos are not in git. To rebuild an approach's comparison videos, run e.g.
`.venv/bin/python -m motion_studio.approaches.a3_multilayer_parallax`.
