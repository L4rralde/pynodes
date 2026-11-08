import numpy as np
import torch

# Min aspect ratio is 0.5. Max is 2.0

def crop_to_target_aspect_ratio(
    img: np.ndarray,
    target_aspect_ratio: float
):
    h, w = img.shape[:2]
    aspect_ratio = h/w

    assert 0.5 < aspect_ratio < 2.0

    if abs(aspect_ratio - target_aspect_ratio) < 1e-2:
        return img

    if target_aspect_ratio < aspect_ratio:
        target_h = int(round(target_aspect_ratio * w))
        low = (h - target_h)//2
        return img[low: low+target_h, :, :]
    else:
        target_w = int(round(h / target_aspect_ratio))
        left = (w - target_w)//2
        return img[:, left: left+target_w, :]

