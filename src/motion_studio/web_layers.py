"""Export transparent bg/mid/fg PNG layers for the web parallax playground."""

import os

import cv2
import numpy as np

from motion_studio.layers import split_layers


def export_layers_for_web(img_path, depth_path, out_dir, max_side=800):
    os.makedirs(out_dir, exist_ok=True)
    img = cv2.imread(str(img_path))
    depth = cv2.imread(str(depth_path), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0

    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    size = (int(w * scale), int(h * scale))
    img_s = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    layers = split_layers(img_s, cv2.resize(depth, size, interpolation=cv2.INTER_AREA))

    to8 = lambda a: (np.clip(a, 0, 1) * 255).astype(np.uint8)  # noqa: E731
    cv2.imwrite(os.path.join(out_dir, "bg.png"), layers.background)
    cv2.imwrite(os.path.join(out_dir, "mid.png"), np.dstack([img_s, to8(layers.mid_alpha)]))
    fg_alpha = layers.fg_alpha if layers.fg_alpha is not None else np.zeros(img_s.shape[:2], np.float32)
    cv2.imwrite(os.path.join(out_dir, "fg.png"), np.dstack([img_s, to8(fg_alpha)]))
    print(f"Exported layers to {out_dir} (has_fg={layers.fg_alpha is not None})")


def main():
    """Export bg/mid/fg PNG layers of the four reference scenes for the web gallery."""
    from motion_studio.paths import SAMPLE_NAMES, WEB_STATIC_DIR, sample_depth, sample_image

    for name in SAMPLE_NAMES:
        export_layers_for_web(sample_image(name), sample_depth(name), WEB_STATIC_DIR / "layers" / name)


if __name__ == "__main__":
    main()
