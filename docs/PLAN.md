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

## Phase 1: one engine that makes YouTube-ready scene clips — done

- New scene engine (`scene.py`) built from small parts: camera moves, layer split, particles, light, encoder.
  One shared layer-splitting function (`layers.py`).
- One-way documentary moves of any duration (push in/out, pan, tilt) plus seamless loops (drift, breathe).
  Each plane moves by a different amount, so you get 2.5D depth without warping the art. Extra zoom is
  calculated per move so the frame edge never shows (covered by tests).
- 1920×1080 output: 16:9-ish images fill the frame (centred on the subject); portrait images are shown whole on
  a blurred, shaded backdrop.
- Looks: 8 presets, 5 particle types (dust, embers, fireflies, snow, rain), warm light pulse and fire flicker.
- Speed: ~1.6–1.9 s per second of video for portrait images, ~2.7 s for full-frame 16:9 (the old engine needed
  ~8 s at full frame). A 10-minute episode is about 20–30 minutes. `--draft` previews run faster than real time.
- Depth and layer splits are cached; 56 tests including a golden frame.

Known limits, for Phase 3: the fill behind subjects is soft, so strong pans show a smudge next to characters;
small images upscaled to 1080p look soft.

## Phase 2: the episode pipeline — done

Workflow: write `script.md` → `plan` (shot list for image generation) → record narration → make images →
`build --draft` → `build`.

- **Script format:** one paragraph per image, `##` chapters, `> Image:` notes for the shot list.
- **Voice clean-up** (FFmpeg): rumble cut, non-local-means denoise (removed ~19 dB of hiss in tests vs ~4 dB for
  the first choice), EQ, de-esser, compressor, limiter. Can be switched off if you clean the voice elsewhere.
- **Timing:** faster-whisper `base` on CPU (~7 s per minute of audio) + word alignment to the script: 96-97% of
  words matched on a test narration; unmatched words (names, numbers) are interpolated. Cuts land 0.25 s before
  each paragraph, with a 1.5 s minimum per image.
- **Video:** one cached segment per scene; crossfades are split across neighbouring segments so they join
  seamlessly; only changed scenes are re-rendered. Missing images become placeholder cards in drafts.
- **Audio:** music loops, fades in/out and ducks ~10 dB under the voice; two-pass loudness to -14 LUFS.
- **Exports:** video, `subtitles.srt` (script spelling), `chapters.txt` (checked against YouTube's rules),
  `timeline.json`.
- Measured: 56 s test episode at 1080p in 147 s from scratch (≈2.6 s per second of video → a 15-minute episode
  in ~35-40 min); after changing one scene, 50 s.

Ideas for later: a stronger AI voice enhancer (e.g. DeepFilterNet) as an option; burned-in subtitles for Shorts;
rendering scenes on both CPU cores in parallel.

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

- Answered: 10-15 minute episodes; script first, then voice, then images from the script; music maybe.
- Will images be generated at 16:9 from now on?
