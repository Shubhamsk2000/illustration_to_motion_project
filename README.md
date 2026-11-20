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

## Motion Studio (the web app)

```bash
.venv/bin/motion-studio            # then open http://127.0.0.1:8010
```

Everything below can be done in the browser: create an episode, write the script, upload your narration and music,
drop an image onto each scene, change any scene's look from dropdowns, preview a single scene in a few seconds,
build a draft or the final video (with progress and cancel), then watch it and download the video, subtitles and
chapters. On a remote machine, forward port 8010 (VS Code does this from the Ports panel).

## Make an episode from the command line

```bash
.venv/bin/motion-episode new episodes/roman-soldier --title "A Day in the Life of a Roman Soldier"
#  1. write episodes/roman-soldier/script.md   (one paragraph = one image, "## Heading" = chapter,
#                                               "> Image: ..." = what to draw, not narrated)
.venv/bin/motion-episode plan episodes/roman-soldier    # -> shotlist.md: the image to make for each scene
#  2. record narration.wav following the script, put it in the folder
#  3. generate images/001.png, 002.png ... (16:9) from shotlist.md; optionally add music.mp3
.venv/bin/motion-episode voice episodes/roman-soldier   # optional: listen to the cleaned-up voice
.venv/bin/motion-episode build episodes/roman-soldier --draft   # fast preview, cards for missing images
.venv/bin/motion-episode build episodes/roman-soldier           # final 1080p
```

`build` writes to `export/`: the video (voice cleaned up, music ducked under the voice, -14 LUFS like YouTube
expects), `subtitles.srt` (spelled exactly as in your script) and `chapters.txt` (paste into the description).

How it works: the recording is transcribed with faster-whisper on CPU and matched word by word to the script, so
every image appears just before you start its paragraph. Scenes crossfade into each other. Each scene is cached,
so after changing one scene's look in `episode.yaml` only that scene and its neighbours are re-rendered.

Each scene's look is picked from its words: the `> Image:` note, or the narration if there is no note
(fire/torch/candle/lantern -> campfire, rain/storm -> storm, snow -> winter, battle/siege -> battle,
fog/mist/dawn -> misty, sun/sunset -> sunlit, close-up/face -> portrait, otherwise the default).
`plan` shows the suggestion as `auto_look` for every scene; set `preset:` to choose yourself, or
`auto_look: false` in the settings to turn it off.

`episode.yaml` holds the settings (fps, size, crossfade length, default look, music volume, voice clean-up on/off,
an `image_style` added to every prompt in the shot list) and an optional look per scene, e.g.:

```yaml
- id: '012'
  preset: campfire
  move: pull_out
```

Timing on this machine (2 CPUs): a 15-minute episode takes roughly 35-40 minutes to build the first time; drafts
are several times faster. Episodes live in `episodes/`, which is not committed to git.

## Single clips

```bash
# Every image in a folder -> one 1920x1080 clip each (camera move varies scene to scene)
.venv/bin/motion-batch --input_dir my_scenes --output_dir output/my_episode --duration 7

# Quick 960x540 drafts to pick moves before the real render
.venv/bin/motion-batch --input_dir my_scenes --draft

# One image with a look, overriding parts of it
.venv/bin/motion-batch --image my_scenes/camp.png --preset campfire --move pull_out --duration 8

# Seamless loop
.venv/bin/motion-batch --image my_scenes/forest.png --preset magic_forest --loop --duration 6

```

- **Moves:** `push_in`, `pull_out`, `pan_left`, `pan_right`, `tilt_up`, `tilt_down`, `static`; loops: `drift`, `breathe`.
- **Atmosphere (particles):** `dust`, `embers`, `fireflies`, `snow`, `rain`.
- **Mist:** `fog` (drifts behind the characters), `smoke` (rises over everything).
- **Glow:** `rays` (light shafts from the brightest spot), `firelight` (flickering warm light around the
  brightest flame; flickers the whole picture if no flame is found).
- **Presets:** `documentary`, `portrait`, `epic`, `calm`, `campfire`, `winter`, `storm`, `magic_forest`,
  `misty`, `sunlit`, `battle`.
- **Fit:** 16:9 (or close) images fill the frame; portrait images are shown whole on a blurred backdrop
  (`--fit crop|blur` to force one). Generate images at 16:9, 1920 px wide or more, for the best result.
- Tuning flags: `--strength`, `--depth`, `--breathing`, `--mist`, `--glow`, `--light`, `--fps`, `--size`.
  See `motion-batch -h`.
- The background behind characters is filled in by the LaMa model (downloaded once, ~200 MB, ~5 s per image,
  cached). Without it (offline) a basic fill is used.
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
  effects.py         mist (fog, smoke) and light (rays, firelight)
  models.py          downloaded models (LaMa inpainting), pinned to a tested revision
  depth.py           Depth-Anything-V2 depth with caching
  encode.py          FFmpeg writer
  engine.py          older warp-based preset engine (kept for reference)
  approaches/        the four original experiments (a1..a4)
  batch.py           motion-batch CLI
  episode/           motion-episode: script, voice, timing, subtitles, timeline, mix
  web/               motion-studio: FastAPI app, build jobs, and the editor page (static/)
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
