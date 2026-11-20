"""
FastAPI Server for Illustration Motion Studio & Review Gallery.
Supports side-by-side comparison of all 4 approaches and live interactive 2D website parallax.
"""

import os
import math
import subprocess
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Illustration Motion Studio")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("static", exist_ok=True)
os.makedirs("experiments", exist_ok=True)
os.makedirs("static/custom_renders", exist_ok=True)

app.mount("/experiments", StaticFiles(directory="experiments"), name="experiments")
app.mount("/art", StaticFiles(directory="my_favourite_artstyle"), name="art")
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def index():
    return FileResponse("static/index.html")

@app.get("/api/data")
def get_data():
    images = [
        {
            "id": "warriors",
            "title": "Samurai Warriors in Field",
            "source": "/art/image.png",
            "approaches": {
                "approach1": {
                    "mp4": "/experiments/approach_1_semantic_procedural/warriors_approach1.mp4",
                    "gif": "/experiments/approach_1_semantic_procedural/warriors_approach1.gif",
                },
                "approach2": {
                    "mp4": "/experiments/approach_2_depth_mesh_parallax/warriors_approach2.mp4",
                    "gif": "/experiments/approach_2_depth_mesh_parallax/warriors_approach2.gif",
                },
                "approach3": {
                    "mp4": "/experiments/approach_3_multilayer_2d_parallax/warriors_approach3.mp4",
                    "gif": "/experiments/approach_3_multilayer_2d_parallax/warriors_approach3.gif",
                },
                "approach4": {
                    "mp4": "/experiments/approach_4_subtle_2d_breathing/warriors_approach4.mp4",
                    "gif": "/experiments/approach_4_subtle_2d_breathing/warriors_approach4.gif",
                }
            },
            "layers": {
                "bg": "/static/layers/warriors/bg.png",
                "mid": "/static/layers/warriors/mid.png",
                "fg": "/static/layers/warriors/fg.png"
            }
        },
        {
            "id": "army",
            "title": "Imperial Army with Banners",
            "source": "/art/image copy.png",
            "approaches": {
                "approach1": {
                    "mp4": "/experiments/approach_1_semantic_procedural/army_approach1.mp4",
                    "gif": "/experiments/approach_1_semantic_procedural/army_approach1.gif",
                },
                "approach2": {
                    "mp4": "/experiments/approach_2_depth_mesh_parallax/army_approach2.mp4",
                    "gif": "/experiments/approach_2_depth_mesh_parallax/army_approach2.gif",
                },
                "approach3": {
                    "mp4": "/experiments/approach_3_multilayer_2d_parallax/army_approach3.mp4",
                    "gif": "/experiments/approach_3_multilayer_2d_parallax/army_approach3.gif",
                },
                "approach4": {
                    "mp4": "/experiments/approach_4_subtle_2d_breathing/army_approach4.mp4",
                    "gif": "/experiments/approach_4_subtle_2d_breathing/army_approach4.gif",
                }
            },
            "layers": {
                "bg": "/static/layers/army/bg.png",
                "mid": "/static/layers/army/mid.png",
                "fg": "/static/layers/army/fg.png"
            }
        },
        {
            "id": "king",
            "title": "King in the Throne Hall",
            "source": "/art/image copy 2.png",
            "approaches": {
                "approach1": {
                    "mp4": "/experiments/approach_1_semantic_procedural/king_approach1.mp4",
                    "gif": "/experiments/approach_1_semantic_procedural/king_approach1.gif",
                },
                "approach2": {
                    "mp4": "/experiments/approach_2_depth_mesh_parallax/king_approach2.mp4",
                    "gif": "/experiments/approach_2_depth_mesh_parallax/king_approach2.gif",
                },
                "approach3": {
                    "mp4": "/experiments/approach_3_multilayer_2d_parallax/king_approach3.mp4",
                    "gif": "/experiments/approach_3_multilayer_2d_parallax/king_approach3.gif",
                },
                "approach4": {
                    "mp4": "/experiments/approach_4_subtle_2d_breathing/king_approach4.mp4",
                    "gif": "/experiments/approach_4_subtle_2d_breathing/king_approach4.gif",
                }
            },
            "layers": {
                "bg": "/static/layers/king/bg.png",
                "mid": "/static/layers/king/mid.png",
                "fg": "/static/layers/king/fg.png"
            }
        },
        {
            "id": "fox",
            "title": "Forest Spirit & Kitsune Fox",
            "source": "/art/image copy 3.png",
            "approaches": {
                "approach1": {
                    "mp4": "/experiments/approach_1_semantic_procedural/fox_approach1.mp4",
                    "gif": "/experiments/approach_1_semantic_procedural/fox_approach1.gif",
                },
                "approach2": {
                    "mp4": "/experiments/approach_2_depth_mesh_parallax/fox_approach2.mp4",
                    "gif": "/experiments/approach_2_depth_mesh_parallax/fox_approach2.gif",
                },
                "approach3": {
                    "mp4": "/experiments/approach_3_multilayer_2d_parallax/fox_approach3.mp4",
                    "gif": "/experiments/approach_3_multilayer_2d_parallax/fox_approach3.gif",
                },
                "approach4": {
                    "mp4": "/experiments/approach_4_subtle_2d_breathing/fox_approach4.mp4",
                    "gif": "/experiments/approach_4_subtle_2d_breathing/fox_approach4.gif",
                }
            },
            "layers": {
                "bg": "/static/layers/fox/bg.png",
                "mid": "/static/layers/fox/mid.png",
                "fg": "/static/layers/fox/fg.png"
            }
        }
    ]

    approaches_info = {
        "approach1": {
            "title": "Approach 1: Semantic Procedural Shaders",
            "badge": "From test.txt",
            "motion_type": "Procedural Wind & Sky Drift",
            "description": "Segments semantic elements (Sky, Foliage, Light). Applies wind sway to trees, directional drift to clouds, and subtle sun luminosity pulses. Leaves character silhouettes completely rigid.",
            "pros": "100% stable character art; realistic natural foliage flutter.",
            "cons": "Static characters without bodily motion."
        },
        "approach2": {
            "title": "Approach 2: 2.5D Depth Mesh Parallax",
            "badge": "From test.txt",
            "motion_type": "Continuous Camera Orbit Mesh",
            "description": "Continuous depth warping where a virtual camera executes a figure-8 orbit. Displaces pixels according to Depth-Anything-V2.",
            "pros": "Creates 3D depth perception automatically.",
            "cons": "Can feel like liquid/rubber warping ('water effect') when applied continuously to solid vector lines."
        },
        "approach3": {
            "title": "Approach 3: Multi-Layer 2D Illustration Parallax",
            "badge": "Recommended (Website Parallax)",
            "motion_type": "Discrete Cutout Multiplane Translation",
            "description": "Separates the artwork into discrete 2D cut-out planes (inpainted background, characters/subject, foreground). Strictly applies rigid 2D affine translations (dX, dY) with zero pixel stretching or warping.",
            "pros": "Exact Firewatch / website parallax look. Preserves crisp vector illustration sharpness with ZERO rubber distortion.",
            "cons": "Requires clean silhouette matting and inpainting behind occluded planes."
        },
        "approach4": {
            "title": "Approach 4: Micro-Motion Parallax + Character Breathing",
            "badge": "Subtle Living Storybook",
            "motion_type": "Micro X/Y Parallax + Base-Anchored Breathing",
            "description": "Multi-layer 2D parallax with micro-translations (2-4px) combined with an organic vertical chest breathing scale (1.007x) anchored at the characters' feet, plus synchronized warm ambient radiance.",
            "pros": "Gives characters a gentle heartbeat and breath of life while remaining 100% faithful to the 2D illustration.",
            "cons": "Motion is intentionally subtle (designed for YouTube storytelling loops)."
        }
    }

    return {"images": images, "approaches": approaches_info}

@app.post("/api/render_custom")
def render_custom(
    image_id: str = Form(...),
    bg_x: float = Form(-3.0),
    bg_y: float = Form(-1.5),
    subj_x: float = Form(3.0),
    subj_y: float = Form(1.5),
    breathing: float = Form(0.007),
    duration: float = Form(4.0)
):
    img_map = {
        "warriors": ("my_favourite_artstyle/image.png", "depth_anything_test.png"),
        "army": ("my_favourite_artstyle/image copy.png", "depth_image_copy.png"),
        "king": ("my_favourite_artstyle/image copy 2.png", "depth_image_copy_2.png"),
        "fox": ("my_favourite_artstyle/image copy 3.png", "depth_image_copy_3.png"),
    }
    if image_id not in img_map:
        return JSONResponse({"error": "Unknown image"}, status_code=400)
        
    img_path, depth_path = img_map[image_id]
    out_name = f"custom_{image_id}_{int(duration)}s"
    out_mp4 = f"static/custom_renders/{out_name}.mp4"
    out_gif = f"static/custom_renders/{out_name}.gif"
    
    # Import approach 4 engine with customized parameters
    from experiments.approach_4_subtle_2d_breathing.run_approach_4 import extract_layers_v4
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
    
    fps = 30
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
        bg_dx = bg_x * math.sin(tau)
        bg_dy = bg_y * math.cos(tau)
        M_bg = np.float32([[1, 0, bg_dx - pad], [0, 1, bg_dy - pad]])
        frame_bg = cv2.warpAffine(bg_clean, M_bg, (tw, th), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT).astype(np.float32)
        
        mid_dx = subj_x * math.sin(tau)
        mid_dy = subj_y * math.cos(tau)
        sy = 1.0 + breathing * math.sin(tau)
        sx = 1.0 + (breathing * 0.35) * math.sin(tau)
        
        M_mid = np.float32([
            [sx, 0, (1.0 - sx) * ax + mid_dx - pad],
            [0, sy, (1.0 - sy) * ay + mid_dy - pad]
        ])
        frame_mid = cv2.warpAffine(img_padded, M_mid, (tw, th), flags=cv2.INTER_LINEAR).astype(np.float32)
        alpha_mid_s = cv2.warpAffine(mid_alpha, M_mid, (tw, th), flags=cv2.INTER_LINEAR)[:, :, None]
        
        comp = frame_bg * (1.0 - alpha_mid_s) + frame_mid * alpha_mid_s
        final_frame = np.clip(comp, 0, 255).astype(np.uint8)
        proc.stdin.write(final_frame.tobytes())
        
    proc.stdin.close()
    proc.wait()
    
    subprocess.run([
        "ffmpeg", "-y", "-i", out_mp4,
        "-vf", "fps=10,scale=320:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse",
        out_gif
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    return {
        "mp4": f"/{out_mp4}",
        "gif": f"/{out_gif}",
        "status": "success"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
