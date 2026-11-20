"""
Approach 2: 2.5D Depth Parallax ("Live Wallpaper" Breathing Effect)
As described in test.txt:
- Depth estimation via Depth-Anything-V2
- 2.5D Mesh Displacement & Virtual Camera Trajectory (Figure-8 orbit & slow zoom)
- Occluded edge padding / inpainting
"""

import os
import math
import subprocess
import cv2
import numpy as np

def run_approach_2(img_path, depth_path, out_mp4, out_gif, duration=4.0, fps=30):
    img = cv2.imread(img_path)
    depth = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
    h, w = img.shape[:2]
    scale = 720.0 / h
    th, tw = int(h * scale), int(w * scale)
    tw = tw if tw % 2 == 0 else tw - 1
    th = th if th % 2 == 0 else th - 1
    
    img_s = cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)
    depth_s = cv2.resize(depth, (tw, th), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    
    # Slight blur on depth to ensure smooth continuous 3D camera parallax
    depth_smooth = cv2.GaussianBlur(depth_s, (9, 9), 0)
    
    # Pre-pad image to prevent border tearing during 3D camera orbit
    pad = 30
    img_p = cv2.copyMakeBorder(img_s, pad, pad, pad, pad, cv2.BORDER_REFLECT)
    depth_p = cv2.copyMakeBorder(depth_smooth, pad, pad, pad, pad, cv2.BORDER_REFLECT)
    
    grid_y, grid_x = np.mgrid[0:th, 0:tw].astype(np.float32)
    cx, cy = tw * 0.5, th * 0.5
    
    total_frames = int(duration * fps)
    cmd = [
        "ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{tw}x{th}", "-pix_fmt", "bgr24", "-r", str(fps), "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-preset", "veryfast",
        out_mp4
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Virtual camera orbit parameters
    cam_amp_x = 12.0
    cam_amp_y = 6.0
    zoom_amp = 0.03
    
    for f in range(total_frames):
        tau = 2 * math.pi * (f / total_frames)
        
        # 3D Virtual camera trajectory: smooth Lissajous figure-8
        cam_x = cam_amp_x * math.sin(tau)
        cam_y = cam_amp_y * math.cos(tau)
        zoom = 1.0 + zoom_amp * (0.5 + 0.5 * math.sin(tau))
        
        # Continuous depth displacement based on distance from focal depth (0.5)
        disp_factor = (depth_s - 0.5)
        offset_x = cam_x * disp_factor
        offset_y = cam_y * disp_factor
        
        # Camera zoom center
        target_x = cx + (grid_x - cx) / zoom + offset_x + pad
        target_y = cy + (grid_y - cy) / zoom + offset_y + pad
        
        frame = cv2.remap(img_p, target_x.astype(np.float32), target_y.astype(np.float32),
                          cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        
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
    print(f"[Approach 2] Rendered -> {out_mp4} & {out_gif}")

if __name__ == "__main__":
    os.makedirs("experiments/approach_2_depth_mesh_parallax", exist_ok=True)
    images = [
        ("my_favourite_artstyle/image.png", "depth_anything_test.png", "warriors"),
        ("my_favourite_artstyle/image copy.png", "depth_image_copy.png", "army"),
        ("my_favourite_artstyle/image copy 2.png", "depth_image_copy_2.png", "king"),
        ("my_favourite_artstyle/image copy 3.png", "depth_image_copy_3.png", "fox"),
    ]
    for img_p, depth_p, name in images:
        out_mp4 = f"experiments/approach_2_depth_mesh_parallax/{name}_approach2.mp4"
        out_gif = f"experiments/approach_2_depth_mesh_parallax/{name}_approach2.gif"
        run_approach_2(img_p, depth_p, out_mp4, out_gif)
