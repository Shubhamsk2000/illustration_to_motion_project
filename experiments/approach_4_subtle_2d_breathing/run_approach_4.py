"""
Approach 4: Micro-Motion 2D Parallax with Character Breathing & Ambient Radiance
- Discrete 2D cut-out planes (Background, Subject/Midground, Foreground)
- Micro-translation on X/Y (subtle 2-4px multi-plane parallax)
- Organic character breathing: subtle vertical breathing pulse anchored at base (feet)
- Warm atmospheric light pulse synced with breathing cycle
- 100% stable 2D illustration (Zero liquid/water warping)
"""

import os
import math
import subprocess
import cv2
import numpy as np

def extract_layers_v4(img_s, depth_s):
    th, tw = img_s.shape[:2]
    pad = 40
    img_padded = cv2.copyMakeBorder(img_s, pad, pad, pad, pad, cv2.BORDER_REFLECT)
    depth_padded = cv2.copyMakeBorder(depth_s, pad, pad, pad, pad, cv2.BORDER_REFLECT)
    
    # Bilateral smoothing preserves sharp silhouette edges while smoothing flat planes
    depth_bi = cv2.bilateralFilter(depth_padded, d=9, sigmaColor=75, sigmaSpace=75)
    t_split, _ = cv2.threshold(depth_bi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Check if there is an occluding foreground layer
    fg_depths = depth_bi[depth_bi >= t_split]
    has_foreground = False
    t_fg = 255
    if len(fg_depths) > 0 and (np.percentile(fg_depths, 95) - np.percentile(fg_depths, 20)) > 70:
        cand_t = int(np.percentile(fg_depths, 75))
        sobel = np.abs(cv2.Sobel(depth_bi, cv2.CV_32F, 1, 1, ksize=3))
        b_mask = np.abs(depth_bi.astype(float) - cand_t) < 8
        if sobel[b_mask].mean() > 15.0: # True occlusion edge
            t_fg = cand_t
            has_foreground = True
            
    # Masks
    if has_foreground:
        fg_mask = (depth_bi >= t_fg).astype(np.uint8) * 255
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        fg_alpha = cv2.GaussianBlur(fg_mask, (5, 5), 0).astype(np.float32) / 255.0
        
        mid_mask = ((depth_bi >= t_split) & (depth_bi < t_fg)).astype(np.uint8) * 255
        mid_mask = cv2.morphologyEx(mid_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        mid_alpha = cv2.GaussianBlur(mid_mask, (5, 5), 0).astype(np.float32) / 255.0
        
        occ = (depth_bi >= t_split).astype(np.uint8) * 255
        occ_dil = cv2.dilate(occ, np.ones((25, 25), np.uint8))
        bg_clean = cv2.inpaint(img_padded, occ_dil, 9, cv2.INPAINT_TELEA)
    else:
        fg_alpha = None
        mid_mask = (depth_bi >= t_split).astype(np.uint8) * 255
        mid_mask = cv2.morphologyEx(mid_mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        mid_alpha = cv2.GaussianBlur(mid_mask, (7, 7), 0).astype(np.float32) / 255.0
        
        occ_dil = cv2.dilate(mid_mask, np.ones((25, 25), np.uint8))
        bg_clean = cv2.inpaint(img_padded, occ_dil, 9, cv2.INPAINT_TELEA)

    # Find character base anchor
    mid_pts = cv2.findNonZero(mid_mask)
    if mid_pts is not None:
        pts = mid_pts.reshape(-1, 2)
        anchor_y = float(np.percentile(pts[:, 1], 85))
        anchor_x = float(np.mean(pts[:, 0]))
    else:
        anchor_x, anchor_y = tw * 0.5 + pad, th * 0.75 + pad

    return {
        "pad": pad,
        "th": th,
        "tw": tw,
        "has_foreground": has_foreground,
        "bg_clean": bg_clean,
        "img_padded": img_padded,
        "mid_alpha": mid_alpha,
        "fg_alpha": fg_alpha,
        "anchor_x": anchor_x,
        "anchor_y": anchor_y
    }

def run_approach_4(img_path, depth_path, out_mp4, out_gif, duration=4.0, fps=30):
    img = cv2.imread(img_path)
    depth = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
    h, w = img.shape[:2]
    scale = 720.0 / h
    th, tw = int(h * scale), int(w * scale)
    tw = tw if tw % 2 == 0 else tw - 1
    th = th if th % 2 == 0 else th - 1
    
    img_s = cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)
    depth_s = cv2.resize(depth, (tw, th), interpolation=cv2.INTER_AREA)
    
    layers = extract_layers_v4(img_s, depth_s)
    pad = layers["pad"]
    bg_clean = layers["bg_clean"]
    img_padded = layers["img_padded"]
    mid_alpha = layers["mid_alpha"]
    fg_alpha = layers["fg_alpha"]
    has_foreground = layers["has_foreground"]
    ax, ay = layers["anchor_x"], layers["anchor_y"]
    
    total_frames = int(duration * fps)
    cmd = [
        "ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{tw}x{th}", "-pix_fmt", "bgr24", "-r", str(fps), "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-preset", "veryfast",
        out_mp4
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Subtle, elegant micro-amplitudes
    bg_amp_x, bg_amp_y = -2.5, -1.0
    mid_amp_x, mid_amp_y = 2.5, 1.0
    fg_amp_x, fg_amp_y = 6.0, 2.5
    breath_scale_amp = 0.007 # Very subtle, lifelike chest/body expansion
    
    for f in range(total_frames):
        tau = 2 * math.pi * (f / total_frames)
        
        # 1. Background translation + subtle warm ambient light glow
        bg_dx = bg_amp_x * math.sin(tau)
        bg_dy = bg_amp_y * math.cos(tau)
        M_bg = np.float32([[1, 0, bg_dx - pad], [0, 1, bg_dy - pad]])
        frame_bg = cv2.warpAffine(bg_clean, M_bg, (tw, th), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT).astype(np.float32)
        
        # Atmospheric radiance pulse (+4% warm radiance on inhale)
        glow_factor = 1.0 + 0.04 * (0.5 + 0.5 * math.sin(tau))
        frame_bg[:, :, 1] = frame_bg[:, :, 1] * (1.0 + 0.02 * (0.5 + 0.5 * math.sin(tau)))
        frame_bg[:, :, 2] = frame_bg[:, :, 2] * glow_factor # Red / warm
        
        # 2. Subject Transform with Breathing Scale (anchored at base)
        mid_dx = mid_amp_x * math.sin(tau)
        mid_dy = mid_amp_y * math.cos(tau)
        sy = 1.0 + breath_scale_amp * math.sin(tau)
        sx = 1.0 + (breath_scale_amp * 0.35) * math.sin(tau)
        
        M_mid = np.float32([
            [sx, 0, (1.0 - sx) * ax + mid_dx - pad],
            [0, sy, (1.0 - sy) * ay + mid_dy - pad]
        ])
        frame_mid = cv2.warpAffine(img_padded, M_mid, (tw, th), flags=cv2.INTER_LINEAR).astype(np.float32)
        alpha_mid_s = cv2.warpAffine(mid_alpha, M_mid, (tw, th), flags=cv2.INTER_LINEAR)[:, :, None]
        
        if has_foreground:
            fg_dx = fg_amp_x * math.sin(tau)
            fg_dy = fg_amp_y * math.cos(tau)
            M_fg = np.float32([[1, 0, fg_dx - pad], [0, 1, fg_dy - pad]])
            frame_fg = cv2.warpAffine(img_padded, M_fg, (tw, th), flags=cv2.INTER_LINEAR).astype(np.float32)
            alpha_fg_s = cv2.warpAffine(fg_alpha, M_fg, (tw, th), flags=cv2.INTER_LINEAR)[:, :, None]
            
            comp = frame_bg * (1.0 - alpha_mid_s) + frame_mid * alpha_mid_s
            comp = comp * (1.0 - alpha_fg_s) + frame_fg * alpha_fg_s
        else:
            comp = frame_bg * (1.0 - alpha_mid_s) + frame_mid * alpha_mid_s
            
        final_frame = np.clip(comp, 0, 255).astype(np.uint8)
        proc.stdin.write(final_frame.tobytes())
        
    proc.stdin.close()
    proc.wait()
    
    gif_cmd = [
        "ffmpeg", "-y", "-i", out_mp4,
        "-vf", "fps=10,scale=320:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse",
        out_gif
    ]
    subprocess.run(gif_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"[Approach 4] Rendered -> {out_mp4} & {out_gif}")

if __name__ == "__main__":
    os.makedirs("experiments/approach_4_subtle_2d_breathing", exist_ok=True)
    images = [
        ("my_favourite_artstyle/image.png", "depth_anything_test.png", "warriors"),
        ("my_favourite_artstyle/image copy.png", "depth_image_copy.png", "army"),
        ("my_favourite_artstyle/image copy 2.png", "depth_image_copy_2.png", "king"),
        ("my_favourite_artstyle/image copy 3.png", "depth_image_copy_3.png", "fox"),
    ]
    for img_p, depth_p, name in images:
        out_mp4 = f"experiments/approach_4_subtle_2d_breathing/{name}_approach4.mp4"
        out_gif = f"experiments/approach_4_subtle_2d_breathing/{name}_approach4.gif"
        run_approach_4(img_p, depth_p, out_mp4, out_gif)
