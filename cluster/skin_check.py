"""Checks S1, S2, S3 and S5 of "Smoothing the skin weights" (work/pets/PROTOCOL_CAT.md).
  - S1: edges stretched over 30% on gait_check.py's 200 walk frames (same draw): version 22 (old weights, planted
    paws), version 23 (old weights, walking cycle) and the new weights with the cycle rebuilt on them;
  - S2: down paws sliding 1.5 cm or more on all copy walks, new weights and rebuilt cycle;
  - S3: the recordings: on 300 recording frames, edges stretched over 30% with old and new weights (same bones), and
    how far the new weights move each vertex;
  - S5: a sheet of each leg's three most bent frames (rebuilt cycle): version 23 then the new weights, side and front.
CPU only.
usage: python skin_check.py OLD_PAGE NEW_PAGE IK_DIR OLD_GAIT_DIR NEW_GAIT_DIR OUT_DIR"""
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feet_check import DOWN, SKID, Page, skin  # noqa: E402
from feet_ik import CHAINS, bend, joints_of  # noqa: E402
from home2_transfer import edges_of, lbs, read_blobs, stretch  # noqa: E402
from home_coat import render  # noqa: E402

old_dir, new_dir, ik_dir, old_gait, new_gait, out = sys.argv[1:7]
os.makedirs(out, exist_ok=True)
po, pn = Page(old_dir), Page(new_dir)
im = json.load(open(os.path.join(ik_dir, "feet_ik.json")))
IKW = read_blobs(os.path.join(ik_dir, "feet_ik.bin"), im)["walk"].astype(np.float64)
Jr = np.array(im["rest_joints"], dtype=np.float64)
GO = np.load(os.path.join(old_gait, "gait_bones.npz"))["bones"]
GN = np.load(os.path.join(new_gait, "gait_bones.npz"))["bones"]
walks = [tr for tr in po.tracks if tr["kind"] == "walk"]
recs = [tr for tr in po.tracks if tr["kind"] == "rec"]
rows, base = [], 0
for ti, tr in enumerate(walks):
    rows.append((ti, base, len(tr["bones"])))
    base += len(tr["bones"])
e = edges_of(po.fp)
over = lambda pg, Tb: float((stretch(pg.P, lbs(pg.P, pg.ip, pg.wp, Tb), e) > np.log(1.3)).mean())  # noqa: E731
res = {}

# ---- S1, the same 200 frames as gait_check.py
rng = np.random.default_rng(1)
pick = []
for _ in range(200):
    ti, b0, n = rows[rng.integers(len(rows))]
    pick.append((ti, b0, int(rng.integers(n))))
s = {"version 22": [], "version 23": [], "new weights": []}
for ti, b0, k in pick:
    T = walks[ti]["bones"][k:k + 1].astype(np.float64)
    s["version 22"].append(over(po, bend(T, Jr, IKW[b0 + k:b0 + k + 1])[0]))
    s["version 23"].append(over(po, GO[b0 + k].astype(np.float64)))
    s["new weights"].append(over(pn, GN[b0 + k].astype(np.float64)))
res["S1"] = {k: float(np.mean(v)) for k, v in s.items()}
res["S1"]["new_minus_v22_points"] = 100 * (res["S1"]["new weights"] - res["S1"]["version 22"])
res["S1"]["v23_minus_v22_points"] = 100 * (res["S1"]["version 23"] - res["S1"]["version 22"])
res["S1"]["passes"] = res["S1"]["new_minus_v22_points"] <= 1.0
print("S1", json.dumps(res["S1"]), flush=True)

# ---- S2, all copy walks, new weights and the rebuilt cycle
moves = []
for ti, b0, n in rows:
    tr = walks[ti]
    Tb = GN[b0:b0 + n].astype(np.float64)
    X = skin(Tb, pn.P[pn.selp], pn.ip[pn.selp], pn.wp[pn.selp])
    lowP = np.einsum("nvi,ni->nv", X[:, :len(pn.e3p)], tr["H"][:, 1, :]).max(1)
    L, _ = pn.split(X, pn.e3p, pn.lp)
    paw = np.einsum("nij,nlj->nli", tr["H"], L[:, :4]) + tr["root"][:, None]
    h = (lowP + tr["root"][:, 1])[:, None] - paw[..., 1]
    d = h <= DOWN
    mv = np.linalg.norm((paw[1:] - paw[:-1])[..., [0, 2]], axis=-1)
    moves.append(mv[d[:-1] & d[1:]])
mv = np.concatenate(moves)
res["S2"] = {"down_steps": int(len(mv)), "share_sliding_1.5cm_or_more": float((mv > SKID).mean())}
res["S2"]["passes"] = res["S2"]["share_sliding_1.5cm_or_more"] <= 0.044 + 0.005
print("S2", json.dumps(res["S2"]), flush=True)

# ---- S3, the recordings
rng3 = np.random.default_rng(3)
so, sn, disp = [], [], []
for _ in range(300):
    tr = recs[rng3.integers(len(recs))]
    k = int(rng3.integers(len(tr["bones"])))
    Tb = tr["bones"][k].astype(np.float64)
    Xo, Xn = lbs(po.P, po.ip, po.wp, Tb), lbs(pn.P, pn.ip, pn.wp, Tb)
    so.append(float((stretch(po.P, Xo, e) > np.log(1.3)).mean()))
    sn.append(float((stretch(pn.P, Xn, e) > np.log(1.3)).mean()))
    disp.append(np.linalg.norm(Xn - Xo, axis=1))
disp = np.concatenate(disp)
res["S3"] = {"edges_over_30pct_old": float(np.mean(so)), "edges_over_30pct_new": float(np.mean(sn)),
             "difference_points": 100 * (np.mean(sn) - np.mean(so)),
             "vertex_moved_mm_q50_q99_max": (np.percentile(disp, [50, 99, 100]) * 1000).round(2).tolist()}
res["S3"]["passes"] = bool(res["S3"]["difference_points"] <= 0.2 and np.percentile(disp, 50) <= 0.001 and np.percentile(disp, 99) <= 0.01)
print("S3", json.dumps(res["S3"]), flush=True)

# ---- S6, seams stay shut: copies of a vertex (the same place at rest) stay together, on the S3 frames and 100 walk
# frames of the rebuilt cycle
_, inv6 = np.unique(np.round(pn.P / 1e-6).astype(np.int64), axis=0, return_inverse=True)
inv6 = inv6.reshape(-1)
dup = np.where(np.bincount(inv6)[inv6] > 1)[0]
order = dup[np.argsort(inv6[dup], kind="stable")]
grp = inv6[order]
first = np.r_[True, grp[1:] != grp[:-1]]
lead = np.maximum.accumulate(np.where(first, np.arange(len(order)), 0))
pairs = np.stack([order[lead], order], 1)[~first]
gap = 0.0
rng6 = np.random.default_rng(6)
frames6 = [recs[rng6.integers(len(recs))]["bones"][0] for _ in range(50)]
frames6 += [GN[rng6.integers(len(GN))] for _ in range(100)]
for Tb in frames6:
    Xn = lbs(pn.P, pn.ip, pn.wp, Tb.astype(np.float64))
    gap = max(gap, float(np.linalg.norm(Xn[pairs[:, 0]] - Xn[pairs[:, 1]], axis=1).max()) if len(pairs) else 0.0)
res["S6"] = {"seam_copies": int(len(pairs)), "max_gap_mm": gap * 1000, "passes": gap <= 1e-4}
print("S6", json.dumps(res["S6"]), flush=True)

# ---- S5, the sheet
grey = np.full((8, 8, 3), 0.8, np.float32)
J = joints_of(GN.astype(np.float64), Jr)
tiles, worst = [], {}
for p, (bt, bm, bw, _) in enumerate(CHAINS):
    u, v = J[:, bm] - J[:, bt], J[:, bw] - J[:, bm]
    ang = np.degrees(np.arccos(np.clip((u * v).sum(-1) / np.linalg.norm(u, axis=-1) / np.linalg.norm(v, axis=-1), -1, 1)))
    top = np.argsort(-ang)[:3]
    worst[p] = [(int(f), float(ang[f])) for f in top]
    row = []
    for f in top:
        for pg, Tb in ((po, GO[f]), (pn, GN[f])):
            X = lbs(pg.P, pg.ip, pg.wp, Tb.astype(np.float64))
            row.append(render(X, pg.fp, np.zeros((len(X), 2)), grey, "side", size=200))
            row.append(render(X, pg.fp, np.zeros((len(X), 2)), grey, "front", size=200))
    tiles.append(np.concatenate(row, 1))
Image.fromarray((np.concatenate(tiles, 0) * 255).clip(0, 255).astype(np.uint8)).save(os.path.join(out, "s5_sheet.jpg"), quality=85)
res["S5_sheet_frames"] = worst
json.dump(res, open(os.path.join(out, "skin_check.json"), "w"), indent=1)
