"""The judge without ratings, check T2 (work/pets/PROTOCOL_CAT.md, "Check T2, made exact"): a clip's video against our
own reconstruction of the same clip. The reconstruction's posed frames (body frame, articulation only) are drawn from
the video's estimated view direction and measured like ours; the body's travel comes from the video (the reconstruction
has no ground): the ground's motion is set so that the outline's travel equals the video's over the same raw frames.
Windows: common spans of SPAN raw frames. CPU only.
usage: python judge_recon.py REAL_PAIRS_NPZ SEQ KEPT_JSON CAM_NPY RECON_BIN RECON_JSON OUT_NPZ"""
import json
import os
import sys

import numpy as np
from scipy.ndimage import binary_closing, distance_transform_edt, gaussian_filter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from home2_transfer import read_blobs  # noqa: E402
from judge_feat import NAMES, pair_measures  # noqa: E402
import judge_score as js  # noqa: E402

real_npz, seq, kept_json, cam_npy, rbin, rjson, out = sys.argv[1:8]
BLUR, PXU, NS = float(os.environ.get("BLUR", 0.03)), 900.0, 200000
SPAN, STEP = 24, 12
rng = np.random.default_rng(1)

z = np.load(real_npz, allow_pickle=True)
R, cols, seqs = z["rows"], list(z["cols"]), list(z["seqs"])
c = {n: i for i, n in enumerate(cols)}
S = R[R[:, c["seq"]] == seqs.index(seq)]
kept = json.load(open(kept_json))[seq]
cams = np.load(cam_npy)
assert len(kept) == len(cams) == len(S) + 1, (len(kept), len(cams), len(S))
ok = (S[:, c["ground_how"]] > 0) & np.isfinite(S[:, 4:]).all(1)
a_k, b_k = np.array(kept[:-1]), np.array(kept[1:])

m = json.load(open(rjson))
B = read_blobs(rbin, m)
lo, hi = np.array(m["lo"]), np.array(m["hi"])
pos = lo + (hi - lo) * B["pos"].astype(np.float64) / 65535.0
Fm = B["faces"].astype(np.int64)
raw = np.array(m["raw_frames"])
fa = np.linalg.norm(np.cross(pos[0][Fm[:, 1]] - pos[0][Fm[:, 0]], pos[0][Fm[:, 2]] - pos[0][Fm[:, 0]]), axis=1)
fi = rng.choice(len(Fm), NS, p=fa / fa.sum())
u, v = rng.random(NS), rng.random(NS)
flip = u + v > 1
u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
bc = np.stack([1 - u - v, u, v], 1)


def draw(Xa, Xb):
    """camera coordinates (x right, y down, z depth) -> outline pixels and their motion (as judge_ours.draw)"""
    ua, fl = Xa[:, :2] * PXU, (Xb[:, :2] - Xa[:, :2]) * PXU
    lo_ = np.floor(ua.min(0)) - 12
    ij = np.floor(ua - lo_).astype(np.int64)
    W, H = ij[:, 0].max() + 13, ij[:, 1].max() + 13
    lin = ij[:, 1] * W + ij[:, 0]
    order = np.lexsort((Xa[:, 2], lin))
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
        fg[~closed] = 0
    mask = closed
    if BLUR > 0:
        s = BLUR * np.sqrt(mask.sum())
        # the ground's motion mixes in at the outline; here it is not known yet, so the blur is within the outline
        wsum = gaussian_filter(mask.astype(np.float64), s)
        fg = np.stack([gaussian_filter(fg[..., k], s) / np.maximum(wsum, 1e-6) for k in range(2)], -1)
    r, cc = np.nonzero(mask)
    return np.stack([cc + lo_[0], r + lo_[1]], 1).astype(np.float64), fg[r, cc]


# ---- reconstruction pairs, with the video's travel over the same raw frames
rec_rows = []
for j in range(len(raw) - 1):
    a, b = raw[j], raw[j + 1]
    ov = np.clip(np.minimum(b, b_k) - np.maximum(a, a_k), 0, None).astype(np.float64)
    hit = ov > 0
    if not hit.any() or not ok[hit].all() or ov.sum() < (b - a) - 1e-9:
        continue
    frac = ov[hit] / (b_k[hit] - a_k[hit])
    tv = (S[hit][:, [c["tx"], c["ty"]]] * frac[:, None]).sum(0)  # travel in sizes over [a, b)
    Rc = cams[int(np.argmin(np.abs(np.array(kept) - a)))][:3, :3].astype(np.float64)
    Xa = (bc[:, :, None] * pos[j][Fm[fi]]).sum(1) @ Rc.T
    Xb = (bc[:, :, None] * pos[j + 1][Fm[fi]]).sum(1) @ Rc.T
    xy, fl = draw(Xa, Xb)
    size = np.sqrt(len(xy))
    ground = fl.mean(0) - tv * size
    rec_rows.append([j, a, b] + pair_measures(xy, fl, ground).tolist())
Q = np.array(rec_rows)
print("%s: %d video pairs measured of %d; %d reconstruction pairs of %d" % (seq, ok.sum(), len(S), len(Q), len(raw) - 1), flush=True)
qc = {n: 3 + i for i, n in enumerate(NAMES)}
KEYS = ("tx", "ty", "slip10", "slip25", "leg", "torso", "a_fore", "a_hind", "w", "h")
Xv, Xr, spans = [], [], []
for s0 in range(int(kept[0]), int(kept[-1]) - SPAN + 1, STEP):
    vk = (a_k >= s0) & (b_k <= s0 + SPAN)
    rk = (Q[:, 1] >= s0) & (Q[:, 2] <= s0 + SPAN) if len(Q) else np.zeros(0, bool)
    if vk.sum() < 3 or rk.sum() < 3 or not ok[vk].all() or (b_k[vk] - a_k[vk]).sum() < 0.7 * SPAN:
        continue
    Pv = {n: S[vk][:, c[n]] for n in KEYS}
    Pr = {n: Q[rk][:, qc[n]] for n in KEYS}
    if np.hypot(Pv["tx"], Pv["ty"]).sum() / (b_k[vk] - a_k[vk]).sum() < 0.01:
        continue
    fv, mv = js.window_feats(Pv)
    fr, mr = js.window_feats(Pr)
    if np.isfinite(fv).all() and np.isfinite(fr).all():
        Xv.append(fv)
        Xr.append(fr)
        spans.append([s0, mv["aspect"], mr["aspect"]])
Xv, Xr, spans = np.array(Xv), np.array(Xr), np.array(spans)
np.savez(out, video=Xv, recon=Xr, spans=spans, feats=np.array(js.FEATS))
print("common windows: %d; outline longer-than-tall, video %.2f, reconstruction %.2f (medians)" % (
    len(Xv), np.median(spans[:, 1]) if len(spans) else np.nan, np.median(spans[:, 2]) if len(spans) else np.nan))
for k, n in enumerate(js.FEATS):
    if len(Xv):
        cc_ = np.corrcoef(Xv[:, k], Xr[:, k])[0, 1] if len(Xv) > 2 else np.nan
        print("  %-15s video q50 %.3f   reconstruction q50 %.3f   correlation %.2f" % (n, np.median(Xv[:, k]), np.median(Xr[:, k]), cc_))
