#!/usr/bin/env python3
"""
motion-batch: turn one image or a folder of images into 16:9 scene clips.

Examples:
    motion-batch --input_dir my_scenes --output_dir output/my_episode --duration 7
    motion-batch --image market.png --preset campfire --move push_in --duration 8
    motion-batch --input_dir my_scenes --draft            # quick 960x540 previews to pick moves
    motion-batch --image fox.png --loop --duration 6      # seamless loop
    motion-batch --approach 3 --image fox.png             # old experimental approaches 1-4
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

from motion_studio.atmosphere import KIND_NAMES
from motion_studio.camera import ALL_MOVES, auto_move
from motion_studio.paths import DEPTH_CACHE_DIR, DEPTH_DIR, OUTPUT_DIR, SAMPLES_DIR
from motion_studio.scene import FITS, PRESETS, SceneRenderer, spec_from_preset

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def find_images(input_dir: Path) -> list[Path]:
    return sorted(p for p in input_dir.iterdir() if p.suffix.lower() in SUPPORTED_EXTENSIONS)


# ---- new scene engine ------------------------------------------------------------------------------------

def render_one(image: Path, out: Path, args, index: int) -> dict:
    move = args.move
    if move == "auto":
        move = auto_move(index, loop=args.loop) if not args.preset else None
    spec = spec_from_preset(
        args.preset, move=move, strength=args.strength, depth=args.depth, breathing=args.breathing,
        atmosphere=args.atmosphere, light=args.light, loop=args.loop or None, fit=args.fit,
    )
    size = (960, 540) if args.draft else tuple(int(v) for v in args.size.lower().split("x"))
    renderer = SceneRenderer(image, spec, size=size, fps=args.fps, duration=args.duration)

    def progress(f):
        print(f"\r  rendering {f * 100:5.1f}%", end="", flush=True)

    if args.draft:
        stats = renderer.render(out, progress=progress, crf=23, encoder_preset="ultrafast")
    else:
        stats = renderer.render(out, progress=progress)
    print()
    return stats


def run_scenes(images: list[Path], args) -> None:
    out_dir = Path(args.output_dir)
    total_t0 = time.time()
    report = []
    for i, image in enumerate(images):
        out = Path(args.output) if (args.output and len(images) == 1) else out_dir / f"{image.stem}.mp4"
        print(f"[{i + 1}/{len(images)}] {image.name}")
        stats = render_one(image, out, args, i)
        spec = stats["spec"]
        print(f"  -> {out}  ({spec['move']}, {spec['atmosphere']}, fit={stats['fit']}, "
              f"{stats['render_seconds']}s for {stats['seconds']}s of video)")
        report.append(stats)
    took = time.time() - total_t0
    video = sum(s["seconds"] for s in report)
    print(f"\nDone: {len(report)} clip(s), {video:.1f}s of video in {took:.1f}s ({took / max(video, 1e-6):.2f}s per second)")
    if len(images) > 1:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "render_report.json").write_text(json.dumps(report, indent=2))


# ---- legacy experimental approaches 1-4 ------------------------------------------------------------------

def get_depth_map(img_path) -> str:
    """Path of a depth PNG for the old approaches (bundled for samples, otherwise estimated and cached)."""
    from motion_studio.depth import depth_for_image, file_hash

    img_path = Path(img_path)
    bundled = DEPTH_DIR / f"{img_path.stem}.png"
    if img_path.resolve().parent == SAMPLES_DIR.resolve() and bundled.exists():
        return str(bundled)
    depth_for_image(img_path)  # writes the cache file if needed
    return str(DEPTH_CACHE_DIR / f"{file_hash(img_path)}.png")


def process_single(image_path, output_mp4, output_gif=None, approach=3, duration=4.0, fps=30):
    print(f"\n[Batch] Processing: {image_path}\n  -> Approach {approach}")
    t0 = time.time()
    depth_path = get_depth_map(image_path)
    output_gif = output_gif or os.path.splitext(output_mp4)[0] + ".gif"
    image_path = str(image_path)
    if approach == 1:
        from motion_studio.approaches.a1_semantic_shaders import run_approach_1 as run
    elif approach == 2:
        from motion_studio.approaches.a2_depth_mesh import run_approach_2 as run
    elif approach == 3:
        from motion_studio.approaches.a3_multilayer_parallax import run_approach_3 as run
    elif approach == 4:
        from motion_studio.approaches.a4_breathing import run_approach_4 as run
    else:
        raise ValueError(f"Unknown approach: {approach}. Choose from 1, 2, 3, 4.")
    run(image_path, depth_path, str(output_mp4), str(output_gif), duration=duration, fps=fps)
    print(f"  -> Completed in {time.time() - t0:.2f}s! Saved to: {output_mp4} & {output_gif}")


def run_legacy(images: list[Path], args) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for image in images:
        out = args.output if (args.output and len(images) == 1) else out_dir / f"{image.stem}_approach{args.approach}.mp4"
        process_single(image, out, approach=args.approach, duration=args.duration, fps=args.fps)


# ---- CLI -------------------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Turn illustrations into 16:9 scene clips.",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    src = ap.add_argument_group("input / output")
    src.add_argument("--image", help="a single image")
    src.add_argument("--input_dir", help="a folder of images (default: assets/samples)")
    src.add_argument("--output_dir", default=str(OUTPUT_DIR / "videos"))
    src.add_argument("--output", help="output .mp4 path (single image only)")

    look = ap.add_argument_group("motion")
    look.add_argument("--preset", choices=list(PRESETS), help="a starting look; flags below override it")
    look.add_argument("--move", default="auto", choices=["auto"] + ALL_MOVES,
                      help="camera move; auto varies it from scene to scene")
    look.add_argument("--strength", type=float, help="camera move size (1.0 = normal)")
    look.add_argument("--depth", type=float, help="near/far separation (1.0 = normal)")
    look.add_argument("--breathing", type=float, help="subject breathing amount, 0 = off (default 0.006)")
    look.add_argument("--atmosphere", choices=KIND_NAMES, help="particles")
    look.add_argument("--light", type=float, help="warm light pulse, 0 = off (default 0.03)")
    look.add_argument("--loop", action="store_true", help="seamless loop instead of a one-way move")
    look.add_argument("--fit", choices=FITS, help="crop to fill 16:9, or show the whole image on a blurred backdrop")

    vid = ap.add_argument_group("video")
    vid.add_argument("--duration", type=float, default=6.0, help="seconds per clip")
    vid.add_argument("--fps", type=int, default=24)
    vid.add_argument("--size", default="1920x1080")
    vid.add_argument("--draft", action="store_true", help="fast 960x540 preview")
    vid.add_argument("--approach", type=int, choices=[1, 2, 3, 4],
                     help="use an old experimental approach instead of the scene engine")
    args = ap.parse_args()

    if args.image:
        images = [Path(args.image)]
        if not images[0].exists():
            sys.exit(f"Error: image '{args.image}' does not exist.")
    else:
        input_dir = Path(args.input_dir or SAMPLES_DIR)
        if not input_dir.is_dir():
            sys.exit(f"Error: input directory '{input_dir}' does not exist.")
        images = find_images(input_dir)
        if not images:
            sys.exit(f"No supported images found in '{input_dir}'.")

    if args.approach:
        run_legacy(images, args)
    else:
        run_scenes(images, args)


if __name__ == "__main__":
    main()
