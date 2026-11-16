"""Second cat at home, coat version (work/pets/PROTOCOL_CAT.md, "Second cat at home"): paint the grey and white cat's
photo coat onto Pikachu's (Agent-to-Sim's) home mesh, so the grey cat at home moves exactly as cleanly as Pikachu.
Its own body shape is not kept. CPU only (numpy, scipy, Pillow).

usage: python home2_coat.py PIKACHU_PAGE_DIR UV_NPZ GREY_TEX OUT_DIR
  PIKACHU_PAGE_DIR: the page's a2s_cat.bin/.json (rest mesh with uv, metres); UV_NPZ: the grey cat's UV mesh;
  GREY_TEX: its photo coat (2048 px, same uv convention: row 0 holds v = 1)
Writes OUT_DIR/a2s_cat_tex_grey.jpg (Pikachu's uv layout) and home2_coat.json.
"""
import json
import os
import sys

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from home2_transfer import icp, read_blobs  # noqa: E402

SIZE, K_NEAR, FILL_ROUNDS = 2048, 4, 12


def fill_and_save(rgb, rc, size, path, rounds=FILL_ROUNDS):
    img = np.zeros((size, size, 3), np.float32)
    w = np.zeros((size, size), np.float32)
    img[rc[:, 0], rc[:, 1]] = rgb
    w[rc[:, 0], rc[:, 1]] = 1
    for _ in range(rounds):
        acc, cnt = np.zeros_like(img), np.zeros_like(w)
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            acc += np.roll(np.roll(img * w[..., None], dr, 0), dc, 1)
            cnt += np.roll(np.roll(w, dr, 0), dc, 1)
        grow = (w == 0) & (cnt > 0)
        img[grow] = acc[grow] / cnt[grow][:, None]
        w[grow] = 1
    Image.fromarray((np.clip(img, 0, 1) * 255).round().astype(np.uint8)).save(path, quality=90)


def rasterise(pos, faces, uv, size):
    """every texel centre inside a uv triangle -> its 3D point on the surface and its image row/column"""
    pts, rcs = [], []
    tuv = uv[faces] * size  # (F,3,2) in pixels; x = u, y = v
    for f in range(len(faces)):
        a, b, c = tuv[f]
        x0, x1 = int(np.floor(min(a[0], b[0], c[0]))), int(np.ceil(max(a[0], b[0], c[0])))
        y0, y1 = int(np.floor(min(a[1], b[1], c[1]))), int(np.ceil(max(a[1], b[1], c[1])))
        xs, ys = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5)
        p = np.stack([xs.ravel(), ys.ravel()], 1)
        d = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
        if abs(d) < 1e-12:
            continue
        l1 = ((b[1] - c[1]) * (p[:, 0] - c[0]) + (c[0] - b[0]) * (p[:, 1] - c[1])) / d
        l2 = ((c[1] - a[1]) * (p[:, 0] - c[0]) + (a[0] - c[0]) * (p[:, 1] - c[1])) / d
        l3 = 1 - l1 - l2
        ok = (l1 >= -1e-6) & (l2 >= -1e-6) & (l3 >= -1e-6)
        if not ok.any():
            continue
        L = np.stack([l1[ok], l2[ok], l3[ok]], 1)
        pts.append(L @ pos[faces[f]])
        col = np.clip(np.floor(p[ok, 0]).astype(int), 0, size - 1)
        row = np.clip(size - 1 - np.floor(p[ok, 1]).astype(int), 0, size - 1)  # row 0 holds v = 1
        rcs.append(np.stack([row, col], 1))
    pts, rcs = np.concatenate(pts), np.concatenate(rcs)
    _, keep = np.unique(rcs[:, 0] * size + rcs[:, 1], return_index=True)
    return pts[keep], rcs[keep]


def sample(img, uv):
    """bilinear colour at uv (N,2) of an image whose row 0 holds v = 1"""
    h, w = img.shape[:2]
    x = np.clip(uv[:, 0] * w - 0.5, 0, w - 1.001)
    y = np.clip((1 - uv[:, 1]) * h - 0.5, 0, h - 1.001)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    fx, fy = (x - x0)[:, None], (y - y0)[:, None]
    return ((1 - fx) * (1 - fy) * img[y0, x0] + fx * (1 - fy) * img[y0, x0 + 1]
            + (1 - fx) * fy * img[y0 + 1, x0] + fx * fy * img[y0 + 1, x0 + 1])


def main():
    page_dir, uv_npz, grey_tex, out = sys.argv[1:5]
    os.makedirs(out, exist_ok=True)
    am = json.load(open(os.path.join(page_dir, "a2s_cat.json")))
    A = read_blobs(os.path.join(page_dir, "a2s_cat.bin"), am)
    P, F, UV = A["pos"].astype(np.float64), A["faces"].astype(np.int64), A["uv"].astype(np.float64)
    z = np.load(uv_npz)
    Q, Fq, UVq = z["pos"].astype(np.float64), z["faces"].astype(np.int64), z["uv"].astype(np.float64)
    tex = np.asarray(Image.open(grey_tex).convert("RGB"), dtype=np.float32) / 255.0

    # rigid fit (one scale, rotation, shift), as in home2_transfer.py
    tree_p = cKDTree(P)
    mp, mq = P.mean(0), Q.mean(0)
    ep = np.linalg.eigh(np.cov((P - mp).T))[1][:, ::-1]
    eq = np.linalg.eigh(np.cov((Q - mq).T))[1][:, ::-1]
    s0 = np.sqrt(((P - mp) ** 2).sum(1).mean() / ((Q - mq) ** 2).sum(1).mean())
    fits = []
    for sg in ([1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]):
        R0 = ep @ np.diag(sg) @ eq.T
        fits.append(icp(Q, P, s0, R0, mp - s0 * R0 @ mq, tree_p))
    s, R, t, fd = min(fits, key=lambda f: f[3])[:4]
    Qa = s * Q @ R.T + t
    print("rigid fit: scale %.4f, mean distance %.1f mm" % (s, 1000 * fd), flush=True)

    # Pikachu's texels -> nearest points of the fitted grey cat -> its coat
    tpos, trc = rasterise(P, F, UV, SIZE)
    print("Pikachu's texels on the surface: %d (%.1f%% of the image)" % (len(tpos), 100 * len(tpos) / SIZE ** 2), flush=True)
    d, i = cKDTree(Qa).query(tpos, k=K_NEAR)
    w = 1.0 / np.maximum(d, 1e-4)
    w /= w.sum(1, keepdims=True)
    rgb = sum(w[:, k:k + 1] * sample(tex, UVq[i[:, k]]) for k in range(K_NEAR))
    fill_and_save(rgb, trc, SIZE, os.path.join(out, "a2s_cat_tex_grey.jpg"))
    lum = rgb @ np.array([0.2126, 0.7152, 0.0722])
    info = {"rigid_fit_mean_distance_mm": 1000 * fd, "texels": int(len(tpos)),
            "texel_to_grey_distance_mm_median": 1000 * float(np.median(d[:, 0])),
            "texel_to_grey_distance_mm_p90": 1000 * float(np.percentile(d[:, 0], 90)),
            "share_light_texels_lum_over_0.6": float((lum > 0.6).mean()),
            "grey_share_light_vertices_lum_over_0.6": float(
                (sample(tex, UVq) @ np.array([0.2126, 0.7152, 0.0722]) > 0.6).mean())}
    json.dump(info, open(os.path.join(out, "home2_coat.json"), "w"), indent=1)
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
