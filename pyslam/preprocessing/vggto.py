import numpy as np


def crop_to_target_aspect_ratio(
    img: np.ndarray,
    target_aspect_ratio: float
):
    h, w = img.shape[:2]
    aspect_ratio = h/w

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


def _balanced_target_shape(aspect_ratio, image_resolution, patch_size):
    token_number = (image_resolution // patch_size) ** 2
    w_patches = np.sqrt(token_number / aspect_ratio)
    h_patches = token_number / w_patches
    w_patches = max(1, int(np.round(w_patches)))
    h_patches = max(1, int(np.round(h_patches)))
    return h_patches * patch_size, w_patches * patch_size


def _max_size_target_shape(aspect_ratio, image_resolution, patch_size):
    if aspect_ratio >= 1.0:
        height = image_resolution
        width = _round_to_patch_multiple(image_resolution / aspect_ratio, patch_size)
    else:
        width = image_resolution
        height = _round_to_patch_multiple(image_resolution * aspect_ratio, patch_size)
    return height, width


def _round_to_patch_multiple(value, patch_size):
    return max(patch_size, int(np.round(float(value) / patch_size)) * patch_size)


def get_target_size(
    img: np.ndarray,
    mode: str='balanced',
    image_resolution: int = 512,
    patch_size: int=16
):
    h, w = img.shape[:2]
    aspect_ratio = h/w
    if mode == "balanced":
        target_h, target_w = _balanced_target_shape(aspect_ratio, image_resolution, patch_size)
    elif mode == "max_size":
        target_h, target_w = _max_size_target_shape(aspect_ratio, image_resolution, patch_size)
    else:
        raise ValueError(f"Mode {mode} not allowed")

    return (target_h, target_w)

