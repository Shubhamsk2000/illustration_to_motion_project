import cv2
import numpy as np
import math
import subprocess

img = cv2.imread("my_favourite_artstyle/image.png")
depth = cv2.imread("depth_anything_test.png", cv2.IMREAD_GRAYSCALE)
h, w = img.shape[:2]

scale = 720.0 / h
th, tw = int(h * scale), int(w * scale)
img_s = cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)
depth_s = cv2.resize(depth, (tw, th), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0

fps = 30
duration = 4.0
total_frames = int(fps * duration)

cmd = [
    "ffmpeg", "-y",
    "-f", "rawvideo",
    "-vcodec", "rawvideo",
    "-s", f"{tw}x{th}",
    "-pix_fmt", "bgr24",
    "-r", str(fps),
    "-i", "-",
    "-c:v", "libx264",
    "-pix_fmt", "yuv420p",
    "-crf", "18",
    "-preset", "veryfast",
    "test_loop.mp4"
]
proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

grid_y, grid_x = np.mgrid[0:th, 0:tw].astype(np.float32)
norm_y = grid_y / th
norm_x = grid_x / tw
grass_mask = np.clip((norm_y - 0.55) / 0.35, 0, 1) * np.clip((depth_s - 0.4) / 0.4, 0, 1)

num_particles = 40
np.random.seed(42)
px = (np.random.rand(num_particles) * tw).tolist()
py = (np.random.rand(num_particles) * th).tolist()
psize = np.random.uniform(1.5, 3.5, num_particles).tolist()
pspeed_y = (np.random.uniform(0.3, 0.8, num_particles) * th).tolist()
pphase = (np.random.rand(num_particles) * 2 * math.pi).tolist()

sun_x, sun_y = tw * 0.85, th * 0.15
dist_to_sun = np.sqrt((grid_x - sun_x)**2 + (grid_y - sun_y)**2)
max_dist = np.sqrt(tw**2 + th**2)
sun_glow_base = np.clip(1.0 - (dist_to_sun / (max_dist * 0.65)), 0, 1)**1.8

print("Rendering frames...")
for f in range(total_frames):
    tau = 2 * math.pi * (f / total_frames)
    
    cam_dx = 10.0 * math.sin(tau)
    cam_dy = 5.0 * math.cos(tau)
    
    depth_factor = (depth_s - 0.5)
    offset_x = cam_dx * depth_factor
    offset_y = cam_dy * depth_factor
    
    wind = 7.0 * np.sin(tau + norm_x * 4.0) * grass_mask
    
    map_x = (grid_x + offset_x + wind).astype(np.float32)
    map_y = (grid_y + offset_y).astype(np.float32)
    
    frame = cv2.remap(img_s, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    
    glow_intensity = 0.12 * math.sin(tau)
    frame = np.clip(frame.astype(np.float32) * (1.0 + glow_intensity * sun_glow_base[:, :, None]), 0, 255).astype(np.uint8)
    
    for i in range(num_particles):
        curr_py = (py[i] - pspeed_y[i] * (f / total_frames)) % th
        curr_px = (px[i] + 12 * math.sin(tau + pphase[i])) % tw
        alpha = math.sin(math.pi * (curr_py / th)) * (0.5 + 0.5 * math.sin(tau + pphase[i]))
        alpha = max(0.0, min(1.0, alpha))
        color = (int(180 * alpha), int(225 * alpha), int(255 * alpha))
        cv2.circle(frame, (int(curr_px), int(curr_py)), int(psize[i]), color, -1, cv2.LINE_AA)
        
    proc.stdin.write(frame.tobytes())

proc.stdin.close()
proc.wait()
print("Success! Generated test_loop.mp4")
