"""Side-view strip of one track with and without the paw steps (feet_step.py), the cat alone, posed as the page does it
(index.html: poseA2S / poseGrey, then addSteps). Fixed camera across the window, so a planted paw stays on one spot.
Diagnostic picture for the protocol's render check; CPU only.

usage: python feet_step_strip.py PAGE_DIR STEP_DIR OUT_PNG TRACK_NAME T0 T1 [CAT]
  CAT: p (Pikachu, default) or g (the grey and white cat)
  IK_NPZ=feet_ik_walk3.npz: "steps on" shows the legs bent at their joints (feet_ik.py) instead of feet_step's offsets
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feet_check import Page  # noqa: E402
from home2_retarget import region_retarget  # noqa: E402
from home2_transfer import lbs, read_blobs  # noqa: E402

LEGS = os.environ.get("LEGS") == "1"  # only the lower 18 cm, larger, with the paw tips marked
TILE_W, TILE_H, COLS = (700, 180, 3) if LEGS else (300, 200, 7)
PAW_COL = np.array([[0.2, 0.6, 1.0], [1.0, 0.25, 0.25], [0.3, 0.9, 0.3], [1.0, 0.9, 0.2]])


def raster(P, F, depth, col, W, H):
    """P (V,2) pixel places, depth (V,), F faces, col (F,3) -> image (H,W,3)"""
    img = np.full((H, W, 3), 0.13)
    zb = np.full((H, W), np.inf)
    for f in np.argsort(depth[F].mean(1)):
        a, b, c = P[F[f]]
        x0, x1 = int(max(np.floor(min(a[0], b[0], c[0])), 0)), int(min(np.ceil(max(a[0], b[0], c[0])), W - 1))
        y0, y1 = int(max(np.floor(min(a[1], b[1], c[1])), 0)), int(min(np.ceil(max(a[1], b[1], c[1])), H - 1))
        if x1 < x0 or y1 < y0:
            continue
        xs, ys = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5)
        d = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
        if abs(d) < 1e-12:
            continue
        l1 = ((b[1] - c[1]) * (xs - c[0]) + (c[0] - b[0]) * (ys - c[1])) / d
        l2 = ((c[1] - a[1]) * (xs - c[0]) + (a[0] - c[0]) * (ys - c[1])) / d
        ok = (l1 >= 0) & (l2 >= 0) & (1 - l1 - l2 >= 0)
        if not ok.any():
            continue
        z = (l1 * depth[F[f, 0]] + l2 * depth[F[f, 1]] + (1 - l1 - l2) * depth[F[f, 2]])[ok]
        yy, xx = (ys[ok] - 0.5).astype(int), (xs[ok] - 0.5).astype(int)
        near = z < zb[yy, xx]
        zb[yy[near], xx[near]] = z[near]
        img[yy[near], xx[near]] = col[f]
    return img


def main():
    page_dir, step_dir, out, name, t0, t1 = sys.argv[1:7]
    cat = sys.argv[7] if len(sys.argv) > 7 else "p"
    t0, t1 = int(t0), int(t1)
    page = Page(page_dir)
    ti = [i for i, tr in enumerate(page.tracks) if tr["name"] == name][0]
    tr = page.tracks[ti]
    IKB = None
    if os.environ.get("IK_NPZ"):  # bent bones from feet_ik.py, by frame of the track
        zz = np.load(os.environ["IK_NPZ"])
        IKB = dict(zip(zz["t"].tolist(), zz["bones"]))
    meta = json.load(open(os.path.join(step_dir, "feet_step.json")))
    S = read_blobs(os.path.join(step_dir, "feet_step.bin"), meta)
    base = sum(len(t["bones"]) for t in page.tracks[:ti] if t["kind"] == tr["kind"])
    offs = S[("rec_" if tr["kind"] == "rec" else "walk_") + cat].astype(np.float64)[base + t0:base + t1 + 1]
    idx, w = S["leg_%s_idx" % cat].astype(np.int64), S["leg_%s_w" % cat].astype(np.float64)
    cnt = np.cumsum([0] + meta["leg_counts"][cat])
    rest, si, sw, F = (page.P, page.ip, page.wp, page.fp) if cat == "p" else (page.Q, page.ig, page.wg, page.fg)
    tips = (page.lp if cat == "p" else page.lg)[:4]
    frames = []
    for k, t in enumerate(range(t0, t1 + 1)):
        T = tr["bones"][t:t + 1].astype(np.float64)
        Hh, r = tr["H"][t], tr["root"][t]
        XP = lbs(page.P, page.ip, page.wp, T[0])
        if cat == "p":
            X = XP
        else:
            dn = Hh[1]
            X = lbs(page.Q, page.ig, page.wg, region_retarget(T, page.mix, page.C)[0])
            X += ((XP[::3] @ dn).max() - (X[::3] @ dn).max()) * dn
        if IKB is not None:  # "on": the legs bent at their joints (feet_ik.py), no vertex offsets
            T1 = IKB[t][None]
            XP1 = lbs(page.P, page.ip, page.wp, T1[0])
            if cat == "p":
                X1 = XP1
            else:
                X1 = lbs(page.Q, page.ig, page.wg, region_retarget(T1, page.mix, page.C)[0])
                X1 += ((XP1[::3] @ Hh[1]).max() - (X1[::3] @ Hh[1]).max()) * Hh[1]
        else:
            X1 = X.copy()
            for p in range(4):
                v = idx[cnt[p]:cnt[p + 1]]
                X1[v] += w[cnt[p]:cnt[p + 1], None] * offs[k, p][None]
        frames.append([X @ Hh.T + r, X1 @ Hh.T + r])
    # fixed camera: looking across the cat's mean heading, the tail to the right, the room's down down
    Hs = tr["H"][t0:t1 + 1]
    fwd = Hs[:, :, 2].mean(0)
    fwd[1] = 0
    fwd /= np.linalg.norm(fwd)
    lat = np.cross(fwd, [0, 1, 0])
    allX = np.concatenate([x for fr in frames for x in fr])
    u, v = allX @ fwd, allX[:, 1]
    lo, hi = np.array([u.min(), v.min()]), np.array([u.max(), v.max()])
    if LEGS:
        lo[1] = hi[1] - 0.18
    sc = min((TILE_W - 10) / (hi[0] - lo[0]), (TILE_H - 24) / (hi[1] - lo[1]))
    n = len(frames)
    rows = (n + COLS - 1) // COLS
    sheet = np.full((rows * 2 * TILE_H, COLS * TILE_W, 3), 0.05)
    light = np.array([0.3, -0.8, 0.5])
    light /= np.linalg.norm(light)
    base_col = np.array([0.93, 0.62, 0.30]) if cat == "p" else np.array([0.72, 0.72, 0.75])
    for k, fr in enumerate(frames):
        for j, X in enumerate(fr):
            P = np.c_[(X @ fwd - lo[0]) * sc + 5, (X[:, 1] - lo[1]) * sc + 19]
            nrm = np.cross(X[F[:, 1]] - X[F[:, 0]], X[F[:, 2]] - X[F[:, 0]])
            nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
            col = base_col * (0.35 + 0.65 * np.abs(nrm @ light))[:, None]
            img = raster(P, F, -(X @ lat), col, TILE_W, TILE_H)
            low = int((X[:, 1].max() - lo[1]) * sc + 19)
            if 0 <= low < TILE_H:
                img[low, :, :] = 0.55
            step = int(round(0.05 * sc)) if LEGS else 50  # fixed marks (every 5 cm with LEGS), so a planted paw can be seen not to move
            for x in range(0, TILE_W, max(step, 2)):
                img[TILE_H - 6:, x, :] = 0.8
            if LEGS:
                for p, s_ in enumerate(tips):
                    c = X[s_].mean(0)
                    px, py = int((c @ fwd - lo[0]) * sc + 5), int((c[1] - lo[1]) * sc + 19)
                    img[max(py - 3, 0):py + 4, max(px - 3, 0):px + 4] = PAW_COL[p]
            rr, cc = k // COLS, k % COLS
            sheet[(2 * rr + j) * TILE_H:(2 * rr + j + 1) * TILE_H, cc * TILE_W:(cc + 1) * TILE_W] = img
        print("frame %d" % (t0 + k), flush=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(sheet.shape[1] / 100, sheet.shape[0] / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(np.clip(sheet, 0, 1))
    ax.axis("off")
    for k in range(n):
        rr, cc = k // COLS, k % COLS
        for j, lab in enumerate(("steps off", "steps on")):
            ax.text(cc * TILE_W + 4, (2 * rr + j) * TILE_H + 13, "%s  %.1f s  %s" % (name, (t0 + k) / 10, lab),
                    color="w", fontsize=8)
    fig.savefig(out)


if __name__ == "__main__":
    main()
