"""fill_and_save from cat_uv_bake.py, importable without open3d (for env-lab4d)."""
import numpy as np


def fill_and_save(rgb_texels, rc, size, path, rounds=12):
    """paint the colours into the image, spread them a few pixels into the empty space (so seams do not show), save"""
    from PIL import Image
    img = np.zeros((size, size, 3), np.float32)
    w = np.zeros((size, size), np.float32)
    img[rc[:, 0], rc[:, 1]] = rgb_texels
    w[rc[:, 0], rc[:, 1]] = 1
    for _ in range(rounds):
        acc = np.zeros_like(img)
        cnt = np.zeros_like(w)
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            acc += np.roll(np.roll(img * w[..., None], dr, 0), dc, 1)
            cnt += np.roll(np.roll(w, dr, 0), dc, 1)
        grow = (w == 0) & (cnt > 0)
        img[grow] = acc[grow] / cnt[grow][:, None]
        w[grow] = 1
    Image.fromarray((np.clip(img, 0, 1) * 255).round().astype(np.uint8)).save(path, quality=90)
