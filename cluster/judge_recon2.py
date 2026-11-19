"""The judge without ratings, check T2 corrected (work/pets/PROTOCOL_CAT.md, "Check T2, corrected"): a clip's video
against our own reconstruction of it, drawn into the video's own pixels: the reconstruction posed at each kept frame in
the model's own camera (judge_cam_export.py), projected with that frame's intrinsics, cut at the frame's edge, and
measured with the video's own ground motion for the pair. Same pairs as the video (kept frame to next kept frame).
CPU only.
usage: python judge_recon2.py REAL_PAIRS_NPZ(BORDER_OK) REAL_JSON SEQ RECON_CAM_NPZ OUT_NPZ"""
import json
import os
import sys

import numpy as np
from scipy.ndimage import binary_closing, distance_transform_edt, gaussian_filter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from judge_feat import NAMES, pair_measures  # noqa: E402
import judge_score as js  # noqa: E402

real_npz, real_json, seq, cam_npz, out = sys.argv[1:6]
BLUR, NS, SIZE = float(os.environ.get("BLUR", 0.03)), 200000, 180.0
rng = np.random.default_rng(1)
z = np.load(real_npz, allow_pickle=True)
R, cols, seqs = z["rows"], list(z["cols"]), list(z["seqs"])
c = {n: i for i, n in enumerate(cols)}
S = R[R[:, c["seq"]] == seqs.index(seq)]
Wimg, Himg = [m for m in json.load(open(real_json)) if m["seq"] == seq][0]["image_wh"]
q = np.load(cam_npz)
pos, Fm, ks, raw = q["pos"].astype(np.float64), q["faces"].astype(np.int64), q["ks"].astype(np.float64), q["raw"]
assert len(pos) == len(S) + 1, (len(pos), len(S))
fa = np.linalg.norm(np.cross(pos[0][Fm[:, 1]] - pos[0][Fm[:, 0]], pos[0][Fm[:, 2]] - pos[0][Fm[:, 0]]), axis=1)
fi = rng.choice(len(Fm), NS, p=fa / fa.sum())
u, v = rng.random(NS), rng.random(NS)
flip = u + v > 1
u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
bc = np.stack([1 - u - v, u, v], 1)


def project(k):
    X = (bc[:, :, None] * pos[k][Fm[fi]]).sum(1)
    return np.stack([ks[k, 0] * X[:, 0] / X[:, 2] + ks[k, 2], ks[k, 1] * X[:, 1] / X[:, 2] + ks[k, 3]], 1), X[:, 2]


def draw(pa, za, pb, ground, sc):
    """raw-pixel positions at the two frames (and depth at the first) -> outline pixels and motion, in pixels / sc"""
    ua, fl = pa / sc, (pb - pa) / sc
    lo_ = np.floor(ua.min(0)) - 12
    ij = np.floor(ua - lo_).astype(np.int64)
    W, H = ij[:, 0].max() + 13, ij[:, 1].max() + 13
    lin = ij[:, 1] * W + ij[:, 0]
    order = np.lexsort((za, lin))
    first = np.r_[True, lin[order][1:] != lin[order][:-1]]
    sel = order[first]
    mask = np.zeros((H, W), bool)
    fg = np.zeros((H, W, 2))
    mask[ij[sel, 1], ij[sel, 0]] = True
    fg[ij[sel, 1], ij[sel, 0]] = fl[sel]
    closed = binary_closing(mask, iterations=1)
    if (closed & ~mask).any():
        idx = distance_transform_edt(~mask, return_distances=False, return_indices=True)
        fg = fg[idx[0], idx[1]]
    mask = closed
    g = ground / sc
    fg = np.where(mask[..., None], fg, g)  # outside the outline the image moves as the ground does
    if BLUR > 0:
        s = BLUR * np.sqrt(mask.sum())
        fg = np.stack([gaussian_filter(fg[..., k], s, mode="nearest") for k in range(2)], -1)
    r, cc = np.nonzero(mask)
    xy = np.stack([cc + lo_[0], r + lo_[1]], 1).astype(np.float64)
    inside = (xy[:, 0] * sc >= 0) & (xy[:, 0] * sc < Wimg) & (xy[:, 1] * sc >= 0) & (xy[:, 1] * sc < Himg)  # the frame's edge
    return xy[inside], fg[r, cc][inside], g


rows, iou_like = [], []
ok = (S[:, c["ground_how"]] > 0) & np.isfinite(S[:, 4:]).all(1)
for k in range(len(S)):
    if not ok[k]:
        rows.append([np.nan] * len(NAMES))
        continue
    pa, za = project(k)
    pb, _ = project(k + 1)
    sc = S[k, c["size_px"]] / SIZE
    xy, fl, g = draw(pa, za, pb, S[k, [c["gx"], c["gy"]]], sc)
    if len(xy) < 400:
        rows.append([np.nan] * len(NAMES))
        continue
    m = pair_measures(xy, fl, g)
    rows.append(m.tolist())
    iou_like.append(m[NAMES.index("size_px")] * sc / S[k, c["size_px"]])  # the drawn outline's size against the video's
Q = np.array(rows)
okq = np.isfinite(Q).all(1)
print("%s: video pairs measured %d of %d; reconstruction drawn in %d; its outline's size against the video's: median %.2f" % (
    seq, ok.sum(), len(S), okq.sum(), np.median(iou_like) if iou_like else np.nan), flush=True)
qc = {n: i for i, n in enumerate(NAMES)}
KEYS = ("tx", "ty", "slip10", "slip25", "leg", "torso", "a_fore", "a_hind", "w", "h")
Xv, Xr = [], []
k = 0
both = ok & okq
while k + js.NPAIR <= len(S):
    if both[k:k + js.NPAIR].all():
        Pv = {n: S[k:k + js.NPAIR, c[n]] for n in KEYS}
        Pr = {n: Q[k:k + js.NPAIR, qc[n]] for n in KEYS}
        fv, mv = js.window_feats(Pv)
        fr, mr = js.window_feats(Pr)
        if np.hypot(Pv["tx"], Pv["ty"]).sum() / S[k:k + js.NPAIR, c["gap"]].sum() >= 0.01 and np.isfinite(fv).all() and np.isfinite(fr).all():
            Xv.append(fv)
            Xr.append(fr)
        k += js.HOP
    else:
        k += 1
Xv, Xr = np.array(Xv), np.array(Xr)
np.savez(out, video=Xv, recon=Xr, feats=np.array(js.FEATS))
print("common windows: %d" % len(Xv))
for kk, n in enumerate(js.FEATS):
    if len(Xv):
        cc_ = np.corrcoef(Xv[:, kk], Xr[:, kk])[0, 1] if len(Xv) > 2 else np.nan
        print("  %-15s video q50 %.3f   reconstruction q50 %.3f   correlation %.2f" % (n, np.median(Xv[:, kk]), np.median(Xr[:, kk]), cc_))
