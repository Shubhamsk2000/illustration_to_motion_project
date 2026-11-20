# Improvement plan

**Goal:** go from "a folder of AI illustrations + my narration recording" to a finished, upload-ready
1920×1080 YouTube episode with one command, where every illustration feels alive.

## Where we are

- Four motion approaches were prototyped and compared on four sample scenes (`experiments/`). The layered 2D
  ones (3 and 4: cut-out planes, small parallax, breathing characters) keep the illustration crisp and suit the
  art style best. Depth-mesh warping (2) and shaders (1) smear or "liquefy" the art.
- There is also a preset engine (`engine.py`: camera sway/push-in/pan, wind, sun glow, particles), but it uses the
  warping approach, not the layered one.
- Output is 4–6 s **looping** clips at 720p from **portrait** images. A YouTube episode needs 16:9 1080p clips of
  whatever length each narrated beat lasts.
- Speed on this machine (2 CPUs, no GPU): ~3 s to render 1 s of video at 720×1080. A full 1920×1080 frame is
  ~2.7× larger, so a 10-minute episode would take over an hour today.
- The layer-splitting code exists in three near-identical copies (a3, a4, web_layers). Subjects are found by
  thresholding depth, and the hole behind them is filled with OpenCV inpainting, which smears on big subjects.

## Phase 1: one engine that makes YouTube-ready scene clips

1. Merge approaches 3 + 4 and the preset engine into **one layered engine** built from composable effects:
   camera move, layer parallax, breathing, wind, light, particles. One copy of the layer-splitting code.
2. **Documentary camera moves** that run once over any duration (slow push-in, pull-out, pan left/right, tilt)
   in addition to seamless loops. A 7.4 s narrated beat gets a 7.4 s clip.
3. **16:9 1080p output.** Going forward, generate images at 16:9. For portrait images: blurred-fill
   background or a subject-aware crop.
4. **Speed:** benchmark, then vectorise particles, render at 24/25 fps, cache depth/layers per image, render
   scenes in parallel. Target: a 10-minute episode in under 30 minutes.
5. A few golden-frame tests so refactors don't silently change the look.

## Phase 2: the episode pipeline (biggest win for the channel)

1. An episode folder: `episodes/roman-soldier/{scenes/*.png, narration.wav, script.md, episode.yaml}`.
2. **Auto timing:** transcribe the narration on CPU (faster-whisper) and line it up with the script so each
   image gets its start/end time. Manual overrides in `episode.yaml` (per-scene motion, effect, duration).
3. **Assembly:** scene clips + crossfades + your voice + optional music bed ducked under the voice, loudness
   normalised for YouTube (-14 LUFS), plus an `.srt` subtitle file and chapter timestamps.
4. One command: `motion-episode build episodes/roman-soldier` → `final.mp4`, `subtitles.srt`, `chapters.txt`.

## Phase 3: better separation and history-scene effects

1. Proper subject cut-outs with a background-removal / matting model instead of depth thresholds.
2. Better fill behind subjects with a learned inpainting model (LaMa, runs on CPU) so parallax shows no smears.
3. Text-prompted region masks ("sky", "torch flame", "river") so effects land only where they belong.
4. Effect library for history scenes: fire/torch/candle flicker, smoke, drifting fog, rain, snow, embers,
   dust in light beams, cloth/flag sway, water shimmer.
5. Auto-suggest a motion + effect preset per image from its content (night camp → embers + flicker).

## Phase 4: the studio web app as an episode editor

Scene list with thumbnails and timings, pick motion/effects per scene, instant in-browser preview, re-render a
single scene, build the episode, download. Replaces the current comparison gallery.

## Later / optional

- **Shorts:** cut 9:16 shorts from the same scenes (can reuse the youtube_shorts_generator project).
- **AI image-to-video for a few hero shots:** open models need a GPU, so this would be an optional cloud
  step with real cost; decide once the free pipeline is in use.
- Deploy the studio (as done for the Shorts Studio).

## Open questions

- Typical episode length and how many images per minute?
- Will images be generated at 16:9 from now on?
- Is there always a written script for each recording (makes auto timing much more accurate)?
- Mostly one-way documentary moves, or keep loops as an option?
