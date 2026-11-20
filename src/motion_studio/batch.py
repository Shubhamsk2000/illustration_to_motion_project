#!/usr/bin/env python3
"""
Batch Animation Pipeline for Storytelling Videos
Processes an entire directory of illustration images and outputs seamless looping MP4 and GIF animations.

Usage:
    motion-batch --approach 3                                  # all samples in assets/samples -> output/videos
    motion-batch --approach 3 --input_dir my_scenes --output_dir output/my_episode
    motion-batch --approach 4 --image path/to/image.png --output scene_01.mp4
"""

import os
import sys
import argparse
import time
from pathlib import Path
import cv2

from motion_studio.paths import DEPTH_CACHE_DIR, DEPTH_DIR, OUTPUT_DIR, SAMPLES_DIR

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

def get_depth_map(img_path):
    """Returns a depth map path: a bundled one in assets/depth, a cached one, or a fresh estimate."""
    stem = Path(img_path).stem.replace(" ", "_")
    bundled = DEPTH_DIR / f"{stem}.png"
    if bundled.exists():
        return str(bundled)

    DEPTH_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cached = DEPTH_CACHE_DIR / f"{stem}.png"
    if cached.exists():
        return str(cached)

    print(f"  [Depth] Estimating depth with Depth-Anything-V2 for {os.path.basename(img_path)}...")
    from motion_studio.engine import estimate_depth
    img = cv2.imread(img_path)
    d = estimate_depth(img)  # float32 in [0, 1]
    cv2.imwrite(str(cached), (d * 255.0).round().astype("uint8"))
    return str(cached)

def process_single(
    image_path,
    output_mp4,
    output_gif=None,
    approach=3,
    duration=4.0,
    fps=30
):
    print(f"\n[Batch] Processing: {image_path}")
    print(f"  -> Approach {approach}")
    t0 = time.time()
    
    depth_path = get_depth_map(image_path)
    if output_gif is None:
        output_gif = os.path.splitext(output_mp4)[0] + ".gif"
        
    if approach == 1:
        from motion_studio.approaches.a1_semantic_shaders import run_approach_1
        run_approach_1(image_path, depth_path, output_mp4, output_gif, duration=duration, fps=fps)
    elif approach == 2:
        from motion_studio.approaches.a2_depth_mesh import run_approach_2
        run_approach_2(image_path, depth_path, output_mp4, output_gif, duration=duration, fps=fps)
    elif approach == 3:
        from motion_studio.approaches.a3_multilayer_parallax import run_approach_3
        run_approach_3(image_path, depth_path, output_mp4, output_gif, duration=duration, fps=fps)
    elif approach == 4:
        from motion_studio.approaches.a4_breathing import run_approach_4
        run_approach_4(image_path, depth_path, output_mp4, output_gif, duration=duration, fps=fps)
    else:
        raise ValueError(f"Unknown approach: {approach}. Choose from 1, 2, 3, 4.")
        
    elapsed = time.time() - t0
    print(f"  -> Completed in {elapsed:.2f}s! Saved to: {output_mp4} & {output_gif}")

def main():
    parser = argparse.ArgumentParser(description="Batch Illustration-to-Motion Animation Pipeline")
    parser.add_argument("--approach", type=int, default=3, choices=[1, 2, 3, 4],
                        help="1: Semantic Shaders, 2: Depth Mesh Warping, 3: Multi-Layer 2D Parallax (Default/Website), 4: Micro-Parallax + Breathing")
    parser.add_argument("--input_dir", type=str, default=None, help="Directory containing images to process (default: assets/samples)")
    parser.add_argument("--image", type=str, default=None, help="Single image file to process")
    parser.add_argument("--output_dir", type=str, default=str(OUTPUT_DIR / "videos"), help="Directory to save rendered MP4s")
    parser.add_argument("--output", type=str, default=None, help="Output MP4 path for single image mode")
    parser.add_argument("--duration", type=float, default=4.0, help="Loop duration in seconds (default: 4.0)")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second (default: 30)")
    
    args = parser.parse_args()
    
    if args.image:
        out_file = args.output or f"{Path(args.image).stem}_approach{args.approach}.mp4"
        process_single(
            args.image,
            out_file,
            approach=args.approach,
            duration=args.duration,
            fps=args.fps
        )
        return
        
    input_dir = args.input_dir or str(SAMPLES_DIR)
    if not os.path.exists(input_dir):
        print(f"Error: Input directory '{input_dir}' does not exist.")
        sys.exit(1)
        
    os.makedirs(args.output_dir, exist_ok=True)
    
    image_files = [
        os.path.join(input_dir, f) for f in sorted(os.listdir(input_dir))
        if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS
    ]
    
    if not image_files:
        print(f"No supported images found in '{input_dir}'.")
        return
        
    print(f"Found {len(image_files)} image(s) to animate in '{input_dir}' using Approach {args.approach}.")
    total_t0 = time.time()
    
    for idx, img_path in enumerate(image_files, 1):
        stem = Path(img_path).stem.replace(" ", "_")
        out_path = os.path.join(args.output_dir, f"{stem}_approach{args.approach}.mp4")
        print(f"[{idx}/{len(image_files)}] Starting {os.path.basename(img_path)}...")
        process_single(
            img_path,
            out_path,
            approach=args.approach,
            duration=args.duration,
            fps=args.fps
        )
        
    total_elapsed = time.time() - total_t0
    print(f"\n==========================================")
    print(f"Batch processing complete in {total_elapsed:.1f}s!")
    print(f"All animated videos saved in: {os.path.abspath(args.output_dir)}")
    print(f"==========================================")

if __name__ == "__main__":
    main()
