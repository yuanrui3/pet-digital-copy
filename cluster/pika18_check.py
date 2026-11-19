"""The rebuild from this cat's close-up clips, checks R1 and R2 (work/pets/PROTOCOL_CAT.md): the rebuilt cat drawn in
the model's own cameras against each clip's outlines (R1: intersection over union per kept frame), and a sheet of the
walking clips for the eye (R2: the video frame beside the rebuilt cat in plain grey, from the same camera). CPU only.
usage: python pika18_check.py TOOLS_DIR EXPORT_DIR DATABASE_PROCESSED OUT_DIR
  EXPORT_DIR: pika18_export.py's cam_<clip>.npz; DATABASE_PROCESSED: .../database/processed (Annotations, JPEGImages)"""
import glob
import json
import os
import sys
sys.path.insert(0, sys.argv[1])
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_closing
from home_coat import render

exp, base, out = sys.argv[2:5]
os.makedirs(out, exist_ok=True)
WALK = ["cat-85-0071", "cat-85-0072", "cat-85-0073", "cat-85-0066", "cat-pikachu-0-0000"]
NS, SIZE = 200000, 180.0
rng = np.random.default_rng(1)
grey = np.full((8, 8, 3), 0.8, np.float32)
res, tiles = {}, []
for path in sorted(glob.glob(os.path.join(exp, "cam_*.npz"))):
    clip = os.path.basename(path)[4:-4]
    q = np.load(path)
    pos, F, ks, raw = q["pos"].astype(np.float64), q["faces"].astype(np.int64), q["ks"].astype(np.float64), q["raw"]
    A = os.path.join(base, "Annotations", "Full-Resolution", clip)
    M, K = np.load(os.path.join(A, "crop-256.npy")), np.load(os.path.join(A, "crop-256-crop2raw.npy"))
    Kf = np.load(os.path.join(A, "full-256-crop2raw.npy"))
    Wimg, Himg = 256 * Kf[0, 0], 256 * Kf[0, 1]
    if len(M) != len(pos):
        print("skip %s: %d outlines for %d posed frames" % (clip, len(M), len(pos)))
        continue
    fa = np.linalg.norm(np.cross(pos[0][F[:, 1]] - pos[0][F[:, 0]], pos[0][F[:, 2]] - pos[0][F[:, 0]]), axis=1)
    fi = rng.choice(len(F), NS, p=fa / fa.sum())
    u, v = rng.random(NS), rng.random(NS)
    flip = u + v > 1
    u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    bc = np.stack([1 - u - v, u, v], 1)
    ious = []
    for k in range(len(pos)):
        m = M[k, ..., 0] & M[k, ..., 1]
        if m.sum() < 400:
            continue
        X = (bc[:, :, None] * pos[k][F[fi]]).sum(1)
        px = np.stack([ks[k, 0] * X[:, 0] / X[:, 2] + ks[k, 2], ks[k, 1] * X[:, 1] / X[:, 2] + ks[k, 3]], 1)
        sc = np.sqrt(m.sum() * K[k, 0] * K[k, 1]) / SIZE  # raw pixels per grid pixel: the video's cat about SIZE across
        # one grid for both: the box around the two outlines, cut at the frame's edge
        r, c = np.nonzero(m)
        vx, vy = K[k, 0] * c + K[k, 2], K[k, 1] * r + K[k, 3]
        lo = np.floor(np.minimum(px.min(0), [vx.min(), vy.min()]) / sc) - 2
        hi = np.ceil(np.maximum(px.max(0), [vx.max(), vy.max()]) / sc) + 3
        W, H = int(hi[0] - lo[0]), int(hi[1] - lo[1])
        if W * H > 4e6:
            continue
        gm = np.zeros((H, W), bool)
        ij = np.floor(px / sc - lo).astype(int)
        gm[ij[:, 1], ij[:, 0]] = True
        gm = binary_closing(gm, iterations=1)
        gx, gy = np.meshgrid((np.arange(W) + lo[0] + 0.5) * sc, (np.arange(H) + lo[1] + 0.5) * sc)
        cu, cv = np.round((gx - K[k, 2]) / K[k, 0]).astype(int), np.round((gy - K[k, 3]) / K[k, 1]).astype(int)
        inside = (cu >= 0) & (cu < 256) & (cv >= 0) & (cv < 256)
        vm = np.zeros((H, W), bool)
        vm[inside] = m[cv[inside], cu[inside]]
        frame = (gx >= 0) & (gx < Wimg) & (gy >= 0) & (gy < Himg)
        gm &= frame
        ious.append((gm & vm).sum() / max((gm | vm).sum(), 1))
    res[clip] = {"frames": len(ious), "iou_median": float(np.median(ious)) if ious else None,
                 "iou_min": float(np.min(ious)) if ious else None}
    print("%-20s frames %3d  outline overlap median %.3f  min %.3f" % (clip, len(ious), res[clip]["iou_median"] or 0, res[clip]["iou_min"] or 0), flush=True)
    if clip in WALK:  # the sheet: 5 frames of the clip, video above, the rebuilt cat below
        names = sorted(int(f[:5]) for f in os.listdir(os.path.join(base, "JPEGImages", "Full-Resolution", clip)) if f.endswith(".jpg") and f[:5].isdigit())
        row_v, row_r = [], []
        for k in np.linspace(0, len(pos) - 1, 5).round().astype(int):
            im = Image.open(os.path.join(base, "JPEGImages", "Full-Resolution", clip, "%05d.jpg" % names[k])).convert("RGB")
            m = M[k, ..., 0]
            r, c = np.nonzero(m)
            if len(r) < 50:
                continue
            x0, x1 = K[k, 0] * c.min() + K[k, 2], K[k, 0] * c.max() + K[k, 2]
            y0, y1 = K[k, 1] * r.min() + K[k, 3], K[k, 1] * r.max() + K[k, 3]
            side = 1.25 * max(x1 - x0, y1 - y0)
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            cut = im.crop((int(cx - side / 2), int(cy - side / 2), int(cx + side / 2), int(cy + side / 2))).resize((300, 300))
            if not row_v:
                ImageDraw.Draw(cut).text((6, 4), clip[-9:], fill=(255, 255, 0))
            row_v.append(np.asarray(cut, dtype=np.float32) / 255)
            row_r.append(render(pos[k], F, np.zeros((len(pos[k]), 2)), grey, "front", size=300))
        if row_v:
            tiles.append(np.concatenate([np.concatenate(row_v, 1), np.concatenate(row_r, 1)], 0))
json.dump(res, open(os.path.join(out, "r1_overlap.json"), "w"), indent=1)
for n, t in enumerate(tiles):
    Image.fromarray((t * 255).clip(0, 255).astype(np.uint8)).save(os.path.join(out, "r2_sheet_%d.jpg" % n), quality=85)
walk = [res[c]["iou_median"] for c in WALK if c in res and res[c]["iou_median"] is not None]
print("R1: walking clips' and the Body tab clip's medians %s -> %s" % ([round(x, 3) for x in walk], "pass" if walk and min(walk) >= 0.85 else "FAIL"))
