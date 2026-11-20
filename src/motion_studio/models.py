"""Downloaded models, loaded lazily and pinned to the revision that was tested.

All run on CPU and are licensed for commercial use:
- LaMa inpainting (Carve/LaMa-ONNX, Apache-2.0): fills in the background behind cut-out subjects.
"""

import os

import cv2
import numpy as np

LAMA_REPO, LAMA_FILE, LAMA_REVISION = "Carve/LaMa-ONNX", "lama_fp32.onnx", "c3c0c9e468934d62e79c329e35d82dd09ff8c444"
LAMA_SIZE = 512

_lama = None
_lama_failed = False


def _lama_session():
    global _lama, _lama_failed
    if os.environ.get("MOTION_STUDIO_NO_LAMA"):  # used by fast tests
        return None
    if _lama is None and not _lama_failed:
        try:
            import onnxruntime as ort
            from huggingface_hub import hf_hub_download

            path = hf_hub_download(LAMA_REPO, LAMA_FILE, revision=LAMA_REVISION)
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = os.cpu_count() or 2
            _lama = ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])
        except Exception as e:  # offline, or the download failed: fall back to classic inpainting
            print(f"[models] LaMa unavailable ({type(e).__name__}: {e}); using basic fill")
            _lama_failed = True
    return _lama


def lama_inpaint(img_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
    """Fill mask>127 in img (BGR uint8). Returns the filled image at full size, or None if LaMa is unavailable."""
    sess = _lama_session()
    if sess is None:
        return None
    h, w = img_bgr.shape[:2]
    x = cv2.resize(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), (LAMA_SIZE, LAMA_SIZE), interpolation=cv2.INTER_AREA)
    m = cv2.resize(mask, (LAMA_SIZE, LAMA_SIZE), interpolation=cv2.INTER_NEAREST) > 127
    out = sess.run(None, {
        "image": (x.astype(np.float32) / 255.0).transpose(2, 0, 1)[None],
        "mask": m.astype(np.float32)[None, None],
    })[0][0].transpose(1, 2, 0)
    out = cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    return cv2.resize(out, (w, h), interpolation=cv2.INTER_CUBIC)
