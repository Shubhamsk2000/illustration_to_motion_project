#!/usr/bin/env python3
"""
Batch Animation Pipeline for Storytelling Videos
Processes an entire directory of illustration images and outputs seamless looping MP4 and GIF animations.

Usage:
    python batch_animate.py --approach 3 --input_dir my_favourite_artstyle --output_dir output_videos
    python batch_animate.py --approach 4 --image path/to/image.png --output scene_01.mp4
"""

import os
import sys
import argparse
import time
from pathlib import Path
import cv2

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

def get_depth_map(img_path, depth_dir="depth_maps"):
    """Returns depth map path, computing with Depth-Anything-V2 if not cached."""
    os.makedirs(depth_dir, exist_ok=True)
    stem = Path(img_path).stem.replace(" ", "_")
    depth_path = os.path.join(depth_dir, f"{stem}_depth.png")
    
    # Check if a precomputed depth map already exists in workspace
    known_maps = {
        "image": "depth_anything_test.png",
        "image_copy": "depth_image_copy.png",
        "image_copy_2": "depth_image_copy_2.png",
        "image_copy_3": "depth_image_copy_3.png"
    }
    if stem in known_maps and os.path.exists(known_maps[stem]):
        return known_maps[stem]
        
    if os.path.exists(depth_path):
        return depth_path
        
    print(f"  [Depth] Estimating depth with Depth-Anything-V2 for {os.path.basename(img_path)}...")
    from motion_engine import estimate_depth
    img = cv2.imread(img_path)
    d = estimate_depth(img)
    cv2.imwrite(depth_path, d)
    return depth_path

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
        from experiments.approach_1_semantic_procedural.run_approach_1 import run_approach_1
        run_approach_1(image_path, output_mp4, output_gif, duration=duration, fps=fps)
    elif approach == 2:
        from experiments.approach_2_depth_mesh_parallax.run_approach_2 import run_approach_2
        run_approach_2(image_path, depth_path, output_mp4, output_gif, duration=duration, fps=fps)
    elif approach == 3:
        from experiments.approach_3_multilayer_2d_parallax.run_approach_3 import run_approach_3
        run_approach_3(image_path, depth_path, output_mp4, output_gif, duration=duration, fps=fps)
    elif approach == 4:
        from experiments.approach_4_subtle_2d_breathing.run_approach_4 import run_approach_4
        run_approach_4(image_path, depth_path, output_mp4, output_gif, duration=duration, fps=fps)
    else:
        raise ValueError(f"Unknown approach: {approach}. Choose from 1, 2, 3, 4.")
        
    elapsed = time.time() - t0
    print(f"  -> Completed in {elapsed:.2f}s! Saved to: {output_mp4} & {output_gif}")

def main():
    parser = argparse.ArgumentParser(description="Batch Illustration-to-Motion Animation Pipeline")
    parser.add_argument("--approach", type=int, default=3, choices=[1, 2, 3, 4],
                        help="1: Semantic Shaders, 2: Depth Mesh Warping, 3: Multi-Layer 2D Parallax (Default/Website), 4: Micro-Parallax + Breathing")
    parser.add_argument("--input_dir", type=str, default=None, help="Directory containing images to process")
    parser.add_argument("--image", type=str, default=None, help="Single image file to process")
    parser.add_argument("--output_dir", type=str, default="output_videos", help="Directory to save rendered MP4s")
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
        
    input_dir = args.input_dir or "my_favourite_artstyle"
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
