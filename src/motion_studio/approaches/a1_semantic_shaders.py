"""
Approach 1: Semantic Segmentation + Procedural Shaders
As described in docs/original_idea_notes.txt:
- Masks Sky, Water/River, Foliage, Sun
- Element-specific procedural motion (sky drift, sun radial pulse, wind foliage sway, water flow)
"""

import os
import math
import subprocess
import cv2
import numpy as np

def run_approach_1(img_path, depth_path, out_mp4, out_gif, duration=4.0, fps=30):
    img = cv2.imread(img_path)
    depth = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
    h, w = img.shape[:2]
    scale = 720.0 / h
    th, tw = int(h * scale), int(w * scale)
    tw = tw if tw % 2 == 0 else tw - 1
    th = th if th % 2 == 0 else th - 1
    
    img_s = cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)
    depth_s = cv2.resize(depth, (tw, th), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    
    grid_y, grid_x = np.mgrid[0:th, 0:tw].astype(np.float32)
    norm_y = grid_y / th
    norm_x = grid_x / tw
    
    # 1. Sky mask: depth < 0.2 and top half
    sky_mask = np.clip((0.25 - depth_s) / 0.15, 0, 1) * np.clip((0.6 - norm_y) / 0.3, 0, 1)
    
    # 2. Foliage / Grass mask: lower half with high depth or green hue
    hsv = cv2.cvtColor(img_s, cv2.COLOR_BGR2HSV)
    green_mask = cv2.inRange(hsv, (25, 40, 40), (85, 255, 255)).astype(np.float32) / 255.0
    foliage_mask = np.clip((norm_y - 0.5) / 0.3, 0, 1) * np.clip((depth_s - 0.3) / 0.4, 0, 1)
    
    # 3. Sun position & glow mask
    sun_x, sun_y = tw * 0.8, th * 0.15
    dist_sun = np.sqrt((grid_x - sun_x)**2 + (grid_y - sun_y)**2)
    sun_glow = np.clip(1.0 - dist_sun / (np.sqrt(tw**2 + th**2) * 0.65), 0, 1)**1.8
    
    total_frames = int(duration * fps)
    cmd = [
        "ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{tw}x{th}", "-pix_fmt", "bgr24", "-r", str(fps), "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-preset", "veryfast",
        out_mp4
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    
    for f in range(total_frames):
        tau = 2 * math.pi * (f / total_frames)
        
        # Sky slow directional drift
        sky_shift_x = 8.0 * math.sin(tau) * sky_mask
        
        # Foliage gentle harmonic wind sway
        wind_x = 5.0 * np.sin(tau + norm_x * 3.5) * foliage_mask
        
        map_x = (grid_x + sky_shift_x + wind_x).astype(np.float32)
        map_y = grid_y.astype(np.float32)
        
        frame = cv2.remap(img_s, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        
        # Sun subtle luminosity pulse
        pulse = 0.12 * math.sin(tau)
        frame = np.clip(frame.astype(np.float32) * (1.0 + pulse * sun_glow[:, :, None]), 0, 255).astype(np.uint8)
        
        proc.stdin.write(frame.tobytes())
        
    proc.stdin.close()
    proc.wait()
    
    # Convert to GIF
    gif_cmd = [
        "ffmpeg", "-y", "-i", out_mp4,
        "-vf", "fps=10,scale=320:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse",
        out_gif
    ]
    subprocess.run(gif_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"[Approach 1] Rendered -> {out_mp4} & {out_gif}")


def render_samples():
    """Re-render this approach for the four reference scenes into experiments/."""
    from motion_studio.paths import EXPERIMENTS_DIR, SAMPLE_NAMES, sample_depth, sample_image

    out_dir = EXPERIMENTS_DIR / "approach_1_semantic_procedural"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in SAMPLE_NAMES:
        run_approach_1(
            str(sample_image(name)),
            str(sample_depth(name)),
            str(out_dir / f"{name}_approach1.mp4"),
            str(out_dir / f"{name}_approach1.gif"),
        )


if __name__ == "__main__":
    render_samples()
