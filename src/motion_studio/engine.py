"""
Motion Engine: Turns still illustrations into seamless living cinematic loops.
Features:
- Depth-Anything-V2 automatic depth estimation
- 2.5D Depth Parallax & Camera Movements (Push-in, Sway, Pan, Figure-8)
- Multi-frequency Procedural Wind Displacement (Grass, Foliage, Hair)
- Water Ripple & Flow UV Distortion (Rivers, Lakes, Seas)
- Atmospheric Lighting & Sun Glare Breathing
- Floating Cinematic Particle Simulation (Dust Motes, Embers, Fireflies, Rain)
- Seamless Looping MP4 Output (H.264) via FFmpeg
"""

import os
import math
import subprocess
import numpy as np
import cv2
from motion_studio.depth import estimate_depth, get_depth_pipeline  # noqa: F401 (re-exported)

# Pre-defined storytelling presets
PRESETS = {
    "breezy_wilderness": {
        "name": "🌾 Breezy Wilderness",
        "description": "Gentle wind swaying grass/foliage, golden dust motes in sunbeams, subtle camera breathing",
        "parallax_x": 10.0,
        "parallax_y": 5.0,
        "camera_mode": "sway",  # "sway", "push_in", "pan", "still"
        "zoom_amount": 0.02,
        "wind_strength": 8.0,
        "wind_freq": 3.5,
        "wind_zone": "grass",  # "grass", "foliage", "full", "none"
        "sun_pulse": 0.14,
        "sun_pos": [0.85, 0.15], # [x_ratio, y_ratio]
        "particle_type": "dust_motes", # "dust_motes", "embers", "fireflies", "rain", "none"
        "particle_count": 45,
        "water_flow": False
    },
    "dramatic_epic": {
        "name": "⚔️ Dramatic Epic",
        "description": "Slow cinematic push-in, floating war embers, pulsing rim lighting, tense atmosphere",
        "parallax_x": 6.0,
        "parallax_y": 3.0,
        "camera_mode": "push_in",
        "zoom_amount": 0.06,
        "wind_strength": 3.0,
        "wind_freq": 2.0,
        "wind_zone": "foliage",
        "sun_pulse": 0.08,
        "sun_pos": [0.5, 0.2],
        "particle_type": "embers",
        "particle_count": 35,
        "water_flow": False
    },
    "mystic_grove": {
        "name": "🦊 Mystic Grove",
        "description": "Dappled sunlight shimmer, floating forest spores/fireflies, gentle organic sway",
        "parallax_x": 8.0,
        "parallax_y": 4.0,
        "camera_mode": "sway",
        "zoom_amount": 0.025,
        "wind_strength": 5.0,
        "wind_freq": 2.5,
        "wind_zone": "foliage",
        "sun_pulse": 0.18,
        "sun_pos": [0.15, 0.2],
        "particle_type": "fireflies",
        "particle_count": 30,
        "water_flow": False
    },
    "river_flow": {
        "name": "🌊 River & Water Flow",
        "description": "Active directional water ripples, sky parallax drift, soft ambient breathing",
        "parallax_x": 12.0,
        "parallax_y": 6.0,
        "camera_mode": "pan",
        "zoom_amount": 0.015,
        "wind_strength": 4.0,
        "wind_freq": 3.0,
        "wind_zone": "full",
        "sun_pulse": 0.10,
        "sun_pos": [0.5, 0.1],
        "particle_type": "dust_motes",
        "particle_count": 25,
        "water_flow": True
    },
    "subtle_breathing": {
        "name": "🫁 Living Still (Minimal)",
        "description": "Clean 3D parallax depth breathing with zero distortion, ideal for dialogue scenes",
        "parallax_x": 7.0,
        "parallax_y": 4.0,
        "camera_mode": "sway",
        "zoom_amount": 0.015,
        "wind_strength": 0.0,
        "wind_freq": 1.0,
        "wind_zone": "none",
        "sun_pulse": 0.06,
        "sun_pos": [0.5, 0.5],
        "particle_type": "dust_motes",
        "particle_count": 20,
        "water_flow": False
    }
}

def create_particle_system(count, width, height, particle_type="dust_motes", seed=42):
    """Initializes deterministic looping particle parameters."""
    if count <= 0 or particle_type == "none":
        return None
    rng = np.random.RandomState(seed)
    
    px = rng.rand(count) * width
    py = rng.rand(count) * height
    phase = rng.rand(count) * 2 * math.pi
    
    if particle_type == "dust_motes":
        sizes = rng.uniform(2.0, 5.0, count)
        speeds_y = rng.uniform(0.15, 0.45, count) * height # Float upward/drifting
        sway_amp = rng.uniform(8.0, 20.0, count)
        color_rgb = (255, 230, 160) # Warm golden
    elif particle_type == "embers":
        sizes = rng.uniform(1.8, 4.0, count)
        speeds_y = rng.uniform(0.35, 0.75, count) * height # Rise fast
        sway_amp = rng.uniform(12.0, 25.0, count)
        color_rgb = (255, 120, 40) # Bright orange-red
    elif particle_type == "fireflies":
        sizes = rng.uniform(2.5, 6.0, count)
        speeds_y = rng.uniform(-0.1, 0.1, count) * height # Hover
        sway_amp = rng.uniform(15.0, 30.0, count)
        color_rgb = (160, 255, 120) # Magical neon green-yellow
    elif particle_type == "rain":
        sizes = rng.uniform(1.0, 2.0, count)
        speeds_y = rng.uniform(1.5, 3.0, count) * height # Fall fast downward
        sway_amp = rng.uniform(2.0, 5.0, count)
        color_rgb = (180, 210, 240)
    else:
        sizes = rng.uniform(2.0, 4.0, count)
        speeds_y = rng.uniform(0.2, 0.5, count) * height
        sway_amp = rng.uniform(10.0, 20.0, count)
        color_rgb = (255, 255, 255)
        
    return {
        "count": count,
        "type": particle_type,
        "px": px,
        "py": py,
        "phase": phase,
        "sizes": sizes,
        "speeds_y": speeds_y,
        "sway_amp": sway_amp,
        "color_bgr": (color_rgb[2], color_rgb[1], color_rgb[0])
    }

def stamp_glowing_mote(frame, x, y, radius, color_bgr, alpha):
    """Draws a soft Gaussian glowing mote on frame with additive blending."""
    h, w = frame.shape[:2]
    ix, iy = int(round(x)), int(round(y))
    if ix < -10 or ix >= w + 10 or iy < -10 or iy >= h + 10 or alpha <= 0.01:
        return
        
    rad = int(max(2, math.ceil(radius * 2.5)))
    x1, x2 = max(0, ix - rad), min(w, ix + rad + 1)
    y1, y2 = max(0, iy - rad), min(h, iy + rad + 1)
    if x2 <= x1 or y2 <= y1:
        return
        
    # Generate local distance grid
    gy, gx = np.ogrid[y1 - iy:y2 - iy, x1 - ix:x2 - ix]
    dist_sq = gx*gx + gy*gy
    sigma = radius * 0.9
    val = np.exp(-0.5 * dist_sq / (sigma * sigma)) * alpha
    
    # Additive blend
    b = float(color_bgr[0]) * val
    g = float(color_bgr[1]) * val
    r = float(color_bgr[2]) * val
    
    patch = frame[y1:y2, x1:x2].astype(np.float32)
    patch[:, :, 0] += b
    patch[:, :, 1] += g
    patch[:, :, 2] += r
    frame[y1:y2, x1:x2] = np.clip(patch, 0, 255).astype(np.uint8)

def render_animation(
    image_path_or_arr,
    depth_map=None,
    output_path="output.mp4",
    preset="breezy_wilderness",
    custom_params=None,
    duration=6.0,
    fps=30,
    target_height=1080,
    target_width=None,
    progress_callback=None
):
    """
    Renders a seamless looping animated illustration MP4.
    """
    # 1. Load image
    if isinstance(image_path_or_arr, str):
        img_bgr = cv2.imread(image_path_or_arr)
        if img_bgr is None:
            raise ValueError(f"Could not read image from {image_path_or_arr}")
    else:
        img_bgr = image_path_or_arr.copy()
        
    orig_h, orig_w = img_bgr.shape[:2]
    
    # 2. Resize to target dimension
    if target_height is not None:
        scale = float(target_height) / orig_h
        th = int(orig_h * scale)
        tw = int(orig_w * scale)
    elif target_width is not None:
        scale = float(target_width) / orig_w
        tw = int(orig_w * scale)
        th = int(orig_h * scale)
    else:
        th, tw = orig_h, orig_w
        
    # Ensure even dimensions for H.264
    tw = tw if tw % 2 == 0 else tw - 1
    th = th if th % 2 == 0 else th - 1
    
    img_scaled = cv2.resize(img_bgr, (tw, th), interpolation=cv2.INTER_AREA)
    
    # 3. Depth map
    if depth_map is None:
        depth_raw = estimate_depth(img_scaled)
    elif isinstance(depth_map, str):
        depth_file = cv2.imread(depth_map, cv2.IMREAD_GRAYSCALE)
        depth_raw = depth_file.astype(np.float32) / 255.0
    else:
        depth_raw = depth_map.copy()
        
    if depth_raw.shape[:2] != (th, tw):
        depth_scaled = cv2.resize(depth_raw, (tw, th), interpolation=cv2.INTER_LINEAR)
    else:
        depth_scaled = depth_raw
        
    depth_scaled = np.clip(depth_scaled, 0.0, 1.0)
    
    # 4. Resolve configuration / preset
    config = PRESETS.get(preset, PRESETS["breezy_wilderness"]).copy()
    if custom_params:
        config.update(custom_params)
        
    total_frames = int(duration * fps)
    
    # 5. Precompute meshgrid and masks
    grid_y, grid_x = np.mgrid[0:th, 0:tw].astype(np.float32)
    norm_y = grid_y / th
    norm_x = grid_x / tw
    
    # Wind zone mask
    wzone = config.get("wind_zone", "grass")
    if wzone == "grass":
        # Bottom area with closer depth
        wind_mask = np.clip((norm_y - 0.52) / 0.38, 0, 1) * np.clip((depth_scaled - 0.35) / 0.45, 0, 1)
    elif wzone == "foliage":
        # Mid-depth organic areas, leaves/sides
        wind_mask = np.clip(np.sin(norm_y * math.pi), 0, 1) * np.clip((depth_scaled - 0.2) / 0.6, 0, 1)
    elif wzone == "full":
        wind_mask = np.ones((th, tw), dtype=np.float32)
    else:
        wind_mask = np.zeros((th, tw), dtype=np.float32)
        
    # Sun glow mask
    sun_x_r, sun_y_r = config.get("sun_pos", [0.85, 0.15])
    sun_x = tw * sun_x_r
    sun_y = th * sun_y_r
    dist_to_sun = np.sqrt((grid_x - sun_x)**2 + (grid_y - sun_y)**2)
    max_diag = np.sqrt(tw**2 + th**2)
    sun_glow_base = np.clip(1.0 - (dist_to_sun / (max_diag * 0.7)), 0, 1)**1.8
    sun_glow_base = sun_glow_base[:, :, None] # Expand for broadcasting
    
    # Particles
    particles = create_particle_system(
        count=config.get("particle_count", 35),
        width=tw,
        height=th,
        particle_type=config.get("particle_type", "dust_motes")
    )
    
    # 6. Initialize FFmpeg subprocess
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
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
        "-preset", "faster",
        output_path
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    
    cam_mode = config.get("camera_mode", "sway")
    parallax_x = config.get("parallax_x", 8.0)
    parallax_y = config.get("parallax_y", 4.0)
    zoom_amt = config.get("zoom_amount", 0.02)
    wind_strength = config.get("wind_strength", 6.0)
    wind_freq = config.get("wind_freq", 3.0)
    sun_pulse = config.get("sun_pulse", 0.12)
    is_water = config.get("water_flow", False)
    
    for f in range(total_frames):
        # tau goes from 0 to 2*pi for seamless harmonic looping
        tau = 2 * math.pi * (f / total_frames)
        
        # Camera trajectory
        if cam_mode == "sway":
            # Smooth gentle Lissajous figure-8
            cdx = parallax_x * math.sin(tau)
            cdy = parallax_y * math.cos(tau)
            current_zoom = 1.0 + zoom_amt * (0.5 + 0.5 * math.sin(tau))
        elif cam_mode == "push_in":
            # Seamless push-in and exhale
            current_zoom = 1.0 + zoom_amt * (0.5 - 0.5 * math.cos(tau))
            cdx = parallax_x * 0.4 * math.sin(tau)
            cdy = parallax_y * 0.4 * math.sin(tau)
        elif cam_mode == "pan":
            # Slow cinematic drift
            cdx = parallax_x * math.sin(tau)
            cdy = parallax_y * 0.3 * math.cos(tau)
            current_zoom = 1.0 + zoom_amt * (0.5 + 0.5 * math.cos(tau))
        else:
            cdx, cdy, current_zoom = 0.0, 0.0, 1.0
            
        # 2.5D Parallax Displacement: relative to center anchor depth (0.5)
        # Closer objects (depth ~ 1.0) move more; distant objects (depth ~ 0.0) move opposite
        depth_disp = (depth_scaled - 0.45)
        offset_x = cdx * depth_disp
        offset_y = cdy * depth_disp
        
        # Procedural wind displacement
        if wind_strength > 0 and wzone != "none":
            # Dual harmonic wave for natural organic sway
            wave1 = np.sin(tau + norm_x * wind_freq)
            wave2 = 0.35 * np.sin(2 * tau + norm_x * wind_freq * 1.8 + norm_y * 2.0)
            wind_x = wind_strength * (wave1 + wave2) * wind_mask
        else:
            wind_x = 0.0
            
        # Water surface flow ripple
        if is_water:
            # Subtle vertical and horizontal harmonic shimmer on bottom/water regions
            water_mask = np.clip((norm_y - 0.6) / 0.3, 0, 1)
            ripple_x = 4.0 * np.sin(3 * tau + norm_x * 8.0 + norm_y * 12.0) * water_mask
            ripple_y = 2.0 * np.cos(3 * tau + norm_x * 6.0) * water_mask
        else:
            ripple_x, ripple_y = 0.0, 0.0
            
        # Zoom scaling transformation centered at image center
        cx, cy = tw * 0.5, th * 0.5
        scaled_grid_x = cx + (grid_x - cx) / current_zoom
        scaled_grid_y = cy + (grid_y - cy) / current_zoom
        
        map_x = (scaled_grid_x + offset_x + wind_x + ripple_x).astype(np.float32)
        map_y = (scaled_grid_y + offset_y + ripple_y).astype(np.float32)
        
        # Warp image with edge reflection
        frame = cv2.remap(img_scaled, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        
        # Sun & atmospheric lighting breath
        if sun_pulse > 0:
            glow_intensity = sun_pulse * math.sin(tau)
            # Multiplicative exposure boost around sun
            frame = np.clip(frame.astype(np.float32) * (1.0 + glow_intensity * sun_glow_base), 0, 255).astype(np.uint8)
            
        # Draw atmospheric particles
        if particles is not None:
            ptype = particles["type"]
            for i in range(particles["count"]):
                # Position looping
                if ptype == "rain":
                    # Falls downward
                    curr_py = (particles["py"][i] + particles["speeds_y"][i] * (f / total_frames)) % th
                else:
                    # Floats upward
                    curr_py = (particles["py"][i] - particles["speeds_y"][i] * (f / total_frames)) % th
                    
                curr_px = (particles["px"][i] + particles["sway_amp"][i] * math.sin(tau + particles["phase"][i])) % tw
                
                # Smooth border fade-in and fade-out
                fade_y = math.sin(math.pi * (curr_py / th))
                fade_t = 0.6 + 0.4 * math.sin(tau + particles["phase"][i])
                alpha = max(0.0, min(1.0, fade_y * fade_t))
                
                if ptype == "rain":
                    # Draw subtle streak
                    p1 = (int(curr_px), int(curr_py))
                    p2 = (int(curr_px - 2), int(curr_py + 10))
                    cv2.line(frame, p1, p2, (200, 220, 240), 1, cv2.LINE_AA)
                else:
                    stamp_glowing_mote(
                        frame,
                        curr_px,
                        curr_py,
                        radius=particles["sizes"][i],
                        color_bgr=particles["color_bgr"],
                        alpha=alpha
                    )
                    
        proc.stdin.write(frame.tobytes())
        
        if progress_callback and (f % 15 == 0 or f == total_frames - 1):
            progress_callback(float(f + 1) / total_frames)
            
    proc.stdin.close()
    proc.wait()
    
    if proc.returncode != 0:
        err = proc.stderr.read().decode('utf-8')
        raise RuntimeError(f"FFmpeg failed with error:\n{err}")
        
    print(f"[MotionEngine] Finished rendering: {output_path} ({total_frames} frames, {duration}s @ {fps}fps)")
    return output_path
