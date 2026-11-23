"""The walking cycle's checks G2 and G3 (work/pets/PROTOCOL_CAT.md, "A built-in walking cycle on the copy's walks"),
with version 22's walks measured the same way beside it.
  - G2, paws in 3D: on all copy walks at the 0.1 s samples, the share of down steps (paw within 2 cm of the cat's
    lowest point at both samples) whose paw moves 1.5 cm or more across the room (feet_fix.slide_share's rule);
  - G3, the skin: on 200 walk frames drawn at random (the same for both), the share of Pikachu's mesh edges stretched
    by over 30% against the rest shape (feet_ik's rule); and a sheet of each leg's three most bent frames, plain grey
    from the side and the front, each beside version 22 at the same frame;
  - leg bend read-outs: the middle joint's bend from straight, and each leg's extension, while its paw is down.
CPU only.
usage: python gait_check.py PAGE_DIR IK_DIR FLOOR_DIR GAIT_DIR OUT_DIR"""
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

page_dir, ik_dir, floor_dir, gait_dir, out = sys.argv[1:6]
os.makedirs(out, exist_ok=True)
pg = Page(page_dir)
im = json.load(open(os.path.join(ik_dir, "feet_ik.json")))
IKW = read_blobs(os.path.join(ik_dir, "feet_ik.bin"), im)["walk"].astype(np.float64)
Jr = np.array(im["rest_joints"], dtype=np.float64)
GB = np.load(os.path.join(gait_dir, "gait_bones.npz"))
walks = [tr for tr in pg.tracks if tr["kind"] == "walk"]
LEGS = ["front -x", "front +x", "hind -x", "hind +x"]


def versions(tr, base):
    n = len(tr["bones"])
    T = tr["bones"].astype(np.float64)
    return {"version 22": bend(T, Jr, IKW[base:base + n]), "walking cycle": GB["bones"][base:base + n].astype(np.float64)}


def paws_and_low(T, H, r):
    sel = pg.selp
    X = skin(T, pg.P[sel], pg.ip[sel], pg.wp[sel])
    down = H[:, 1, :]
    lowP = np.einsum("nvi,ni->nv", X[:, :len(pg.e3p)], down).max(1)
    L, _ = pg.split(X, pg.e3p, pg.lp)
    paw = np.einsum("nij,nlj->nli", H, L[:, :4]) + r[:, None]
    h = (lowP + r[:, 1])[:, None] - paw[..., 1]
    return paw, h


res = {"G2": {}, "G3": {}, "bend": {}}
skids = {"version 22": [], "walking cycle": []}
bends = {"version 22": [[] for _ in range(4)], "walking cycle": [[] for _ in range(4)]}
exts = {"version 22": [[] for _ in range(4)], "walking cycle": [[] for _ in range(4)]}
rows, base = [], 0
for ti, tr in enumerate(walks):
    n = len(tr["bones"])
    for name, T in versions(tr, base).items():
        paw, h = paws_and_low(T, tr["H"], tr["root"])
        d = h <= DOWN
        step = d[:-1] & d[1:]
        mv = np.linalg.norm((paw[1:] - paw[:-1])[..., [0, 2]], axis=-1)
        skids[name].append(mv[step])
        J = joints_of(T, Jr)
        for p, (bt, bm, bw, _) in enumerate(CHAINS):
            u, v = J[:, bm] - J[:, bt], J[:, bw] - J[:, bm]
            ang = np.degrees(np.arccos(np.clip((u * v).sum(-1) / np.linalg.norm(u, axis=-1) / np.linalg.norm(v, axis=-1), -1, 1)))
            bends[name][p].append(ang[d[:, p]])
            exts[name][p].append((np.linalg.norm(J[:, bw] - J[:, bt], axis=-1) / (np.linalg.norm(u, axis=-1) + np.linalg.norm(v, axis=-1)))[d[:, p]])
    rows.append((ti, base, n))
    base += n
for name in skids:
    s = np.concatenate(skids[name])
    res["G2"][name] = {"down_steps": int(len(s)), "share_sliding_1.5cm_or_more": float((s > SKID).mean()),
                       "share_1.5_to_3cm": float(((s > SKID) & (s <= 0.03)).mean()), "median_move_mm": float(np.median(s) * 1000)}
    res["bend"][name] = {LEGS[p]: {"middle_joint_bend_deg_q10_q50_q90_max": np.percentile(np.concatenate(bends[name][p]), [10, 50, 90, 100]).round(1).tolist(),
                                   "extension_q10_q50_q90": np.percentile(np.concatenate(exts[name][p]), [10, 50, 90]).round(3).tolist()}
                         for p in range(4)}
res["G2"]["passes"] = res["G2"]["walking cycle"]["share_sliding_1.5cm_or_more"] <= 0.03
print(json.dumps(res["G2"], indent=1), flush=True)

# ---- G3: edges stretched over 30%, the same 200 frames for both
rng = np.random.default_rng(1)
e = edges_of(pg.fp)
pick = []
for _ in range(200):
    ti, b0, n = rows[rng.integers(len(rows))]
    pick.append((ti, b0, int(rng.integers(n))))
share = {"version 22": [], "walking cycle": []}
worst = [[] for _ in range(4)]
for ti, b0, k in pick:
    tr = walks[ti]
    for name, T in versions({"bones": tr["bones"][k:k + 1]}, b0 + k).items():
        X = lbs(pg.P, pg.ip, pg.wp, T[0])
        share[name].append(float((stretch(pg.P, X, e) > np.log(1.3)).mean()))
res["G3"] = {name: {"edges_over_30pct_mean": float(np.mean(v)), "frames": len(v)} for name, v in share.items()}
res["G3"]["difference_points"] = 100 * (res["G3"]["walking cycle"]["edges_over_30pct_mean"] - res["G3"]["version 22"]["edges_over_30pct_mean"])
res["G3"]["passes_number"] = res["G3"]["difference_points"] <= 1.0
print(json.dumps(res["G3"], indent=1), flush=True)

# ---- the sheet: each leg's three most bent frames (walking cycle), plain grey, from the side and from the front
grey = np.full((8, 8, 3), 0.8, np.float32)
allb = GB["bones"]
J = joints_of(allb.astype(np.float64), Jr)
tiles = []
for p, (bt, bm, bw, _) in enumerate(CHAINS):
    u, v = J[:, bm] - J[:, bt], J[:, bw] - J[:, bm]
    ang = np.degrees(np.arccos(np.clip((u * v).sum(-1) / np.linalg.norm(u, axis=-1) / np.linalg.norm(v, axis=-1), -1, 1)))
    top = np.argsort(-ang)[:3]
    row = []
    for f in top:
        X = lbs(pg.P, pg.ip, pg.wp, allb[f].astype(np.float64))
        ti = max(k for k, (_, b0_, _) in enumerate(rows) if b0_ <= f)
        T22 = versions({"bones": walks[ti]["bones"][f - rows[ti][1]:f - rows[ti][1] + 1]}, f)["version 22"][0]
        X22 = lbs(pg.P, pg.ip, pg.wp, T22)
        for XX in (X, X22):  # the walking cycle, then version 22 at the same frame
            row.append(render(XX, pg.fp, np.zeros((len(XX), 2)), grey, "side", size=200))
            row.append(render(XX, pg.fp, np.zeros((len(XX), 2)), grey, "front", size=200))
    worst[p] = [(int(f), float(ang[f])) for f in top]
    tiles.append(np.concatenate(row, 1))
Image.fromarray((np.concatenate(tiles, 0) * 255).clip(0, 255).astype(np.uint8)).save(os.path.join(out, "g3_sheet.jpg"), quality=85)
res["G3"]["sheet_rows"] = {LEGS[p]: worst[p] for p in range(4)}
json.dump(res, open(os.path.join(out, "gait_check.json"), "w"), indent=1)
print(json.dumps(res["bend"], indent=1))
