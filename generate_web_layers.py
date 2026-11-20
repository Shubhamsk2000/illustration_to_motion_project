"""
Extract and export transparent PNG layers for the web interactive parallax studio.
"""

import os
import cv2
import numpy as np

def export_layers_for_web(img_path, depth_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    img = cv2.imread(img_path)
    depth = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
    
    h, w = img.shape[:2]
    scale = 800.0 / max(h, w)
    th, tw = int(h * scale), int(w * scale)
    img_s = cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)
    depth_s = cv2.resize(depth, (tw, th), interpolation=cv2.INTER_AREA)
    
    depth_bi = cv2.bilateralFilter(depth_s, d=9, sigmaColor=75, sigmaSpace=75)
    t_split, _ = cv2.threshold(depth_bi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    fg_depths = depth_bi[depth_bi >= t_split]
    has_foreground = False
    t_fg = 255
    if len(fg_depths) > 0 and (np.percentile(fg_depths, 95) - np.percentile(fg_depths, 20)) > 70:
        cand_t = int(np.percentile(fg_depths, 75))
        sobel = np.abs(cv2.Sobel(depth_bi, cv2.CV_32F, 1, 1, ksize=3))
        b_mask = np.abs(depth_bi.astype(float) - cand_t) < 8
        if sobel[b_mask].mean() > 15.0:
            t_fg = cand_t
            has_foreground = True
            
    # Clean background
    occ = (depth_bi >= t_split).astype(np.uint8) * 255
    occ_dil = cv2.dilate(occ, np.ones((25, 25), np.uint8))
    bg_clean = cv2.inpaint(img_s, occ_dil, 9, cv2.INPAINT_TELEA)
    cv2.imwrite(os.path.join(out_dir, "bg.png"), bg_clean)
    
    if has_foreground:
        mid_mask = ((depth_bi >= t_split) & (depth_bi < t_fg)).astype(np.uint8) * 255
        mid_mask = cv2.morphologyEx(mid_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        mid_alpha = cv2.GaussianBlur(mid_mask, (5, 5), 0)
        
        mid_rgba = np.dstack([img_s, mid_alpha])
        cv2.imwrite(os.path.join(out_dir, "mid.png"), mid_rgba)
        
        fg_mask = (depth_bi >= t_fg).astype(np.uint8) * 255
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        fg_alpha = cv2.GaussianBlur(fg_mask, (5, 5), 0)
        
        fg_rgba = np.dstack([img_s, fg_alpha])
        cv2.imwrite(os.path.join(out_dir, "fg.png"), fg_rgba)
    else:
        mid_mask = (depth_bi >= t_split).astype(np.uint8) * 255
        mid_mask = cv2.morphologyEx(mid_mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        mid_alpha = cv2.GaussianBlur(mid_mask, (7, 7), 0)
        
        mid_rgba = np.dstack([img_s, mid_alpha])
        cv2.imwrite(os.path.join(out_dir, "mid.png"), mid_rgba)
        
        # Transparent 1x1 dummy for fg
        dummy = np.zeros((th, tw, 4), dtype=np.uint8)
        cv2.imwrite(os.path.join(out_dir, "fg.png"), dummy)
        
    print(f"Exported layers to {out_dir} (has_fg={has_foreground})")

if __name__ == "__main__":
    images = [
        ("my_favourite_artstyle/image.png", "depth_anything_test.png", "warriors"),
        ("my_favourite_artstyle/image copy.png", "depth_image_copy.png", "army"),
        ("my_favourite_artstyle/image copy 2.png", "depth_image_copy_2.png", "king"),
        ("my_favourite_artstyle/image copy 3.png", "depth_image_copy_3.png", "fox"),
    ]
    for img_p, depth_p, name in images:
        export_layers_for_web(img_p, depth_p, f"static/layers/{name}")
