import os
import numpy as np
from PIL import Image


def load_gray(path, max_size=512):
    img = Image.open(path).convert("L")
    w, h = img.size
    if max(w, h) > max_size:
        ratio = max_size / max(w, h)
        img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
    return np.array(img, dtype=np.float64)


def normalize(arr, lo_pct=0.5, hi_pct=99.5):
    arr = arr.astype(np.float64)
    lo = np.percentile(arr, lo_pct)
    hi = np.percentile(arr, hi_pct)
    if hi - lo < 1e-8:
        lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-8:
        return np.zeros_like(arr, dtype=np.uint8)
    return np.clip((arr - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)


def save_image(arr, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if arr.dtype != np.uint8:
        arr = normalize(arr)
    Image.fromarray(arr).save(path)


def list_images(folder):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    files = [f for f in sorted(os.listdir(folder))
             if os.path.splitext(f)[1].lower() in exts]
    return [os.path.join(folder, f) for f in files]
