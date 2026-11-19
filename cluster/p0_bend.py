"""Phase 0, P0-4 (work/pets/PROTOCOL_CAT.md, "Restart, Phase 0"): can Pikachu's learnt leg chains bend? Each leg chain
(the planted-paw solver's top, middle, wrist/hock and paw bones) is bent at its middle joint by 36 and 45 degrees beyond
its pose, about the axis it usually bends about (the recordings' average, in the top bone's frame), the middle, wrist and
paw bones turning together; the mesh follows by its own skinning. Measured on the leg's vertices (most weight on the
chain): the share of edges stretched or squeezed by over 30%, and the girth near the middle joint (mean distance from
the bone axis in 1 cm slices, within 4 cm of the joint) against unbent. In the rest pose and 4 recorded poses; renders
for the eye. CPU only.
usage: python p0_bend.py TOOLS_DIR PAGE_DIR IK_DIR OUT_DIR   (PAGE_DIR: a2s_cat.bin/.json; IK_DIR: feet_ik.json)"""
import json
import os
import sys
sys.path.insert(0, sys.argv[1])
import numpy as np
from PIL import Image
from feet_ik import CHAINS, joints_of
from home_coat import load_page, render
from home2_transfer import edges_of, lbs

page, ik_dir, out = sys.argv[2:5]
os.makedirs(out, exist_ok=True)
am, A = load_page(page, "a2s_cat")
P, F = A["pos"].astype(np.float64), A["faces"].astype(np.int64)
idx, sw = A["skin_idx"].astype(np.int64), A["skin_w"].astype(np.float64)
bones = A["bones"].astype(np.float64).reshape(len(A["bones"]), 25, 3, 4)
Jr = np.array(json.load(open(os.path.join(ik_dir, "feet_ik.json")))["rest_joints"], dtype=np.float64)
E = edges_of(F)
grey = np.full((8, 8, 3), 0.8, np.float32)
UV0 = np.zeros((len(P), 2))
PAWS = ["-x front", "+x front", "-x hind", "+x hind"]


def rot(axis, th):
    a = axis / np.linalg.norm(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


# the axis each chain usually bends about, in its top bone's frame (recordings, weighted by how much it bends)
Jt = joints_of(bones[::5], Jr)
axes = []
for bt, bm, bw, bp in CHAINS:
    u, v = Jt[:, bm] - Jt[:, bt], Jt[:, bw] - Jt[:, bm]
    n = np.cross(u, v)  # length grows with the bend
    Rt = bones[::5, bt, :, :3]
    loc = np.einsum("nji,nj->ni", Rt, n)  # R^T n: in the top bone's rest frame
    axes.append(loc.sum(0) / np.linalg.norm(loc.sum(0)))
# leg vertices: most of their weight on the chain's bones
wsum = np.zeros((len(P), 4))
for p, ch in enumerate(CHAINS):
    for b in ch:
        wsum[:, p] += (sw * (idx == b)).sum(1)
leg_of = np.where(wsum.max(1) > 0.5, wsum.argmax(1), -1)


def bent(T, p, deg):
    """bones T (25,3,4) with chain p's middle, wrist and paw bones turned by deg about its usual axis at the middle joint"""
    bt, bm, bw, bp = CHAINS[p]
    J = joints_of(T[None], Jr)[0]
    axis = T[bt, :, :3] @ axes[p]
    R, c = rot(axis, np.radians(deg)), J[bm]
    T2 = T.copy()
    for b in (bm, bw, bp):
        T2[b, :, :3] = R @ T[b, :, :3]
        T2[b, :, 3] = R @ (T[b, :, 3] - c) + c
    return T2, J


def measure(X0, X1, p, J, J1):
    """X0 unbent, X1 bent (posed); J, J1 their joints; stretch share on the leg's edges, and the girth ratio near the
    middle joint"""
    on = leg_of == p
    e = E[on[E[:, 0]] & on[E[:, 1]]]
    l0 = np.linalg.norm(X0[e[:, 0]] - X0[e[:, 1]], axis=1)
    l1 = np.linalg.norm(X1[e[:, 0]] - X1[e[:, 1]], axis=1)
    over = float((np.abs(np.log(l1 / np.maximum(l0, 1e-9))) > np.log(1.3)).mean())
    bt, bm, bw, bp = CHAINS[p]
    ratios = []
    for seg, (a, b) in ((0, (bt, bm)), (1, (bm, bw))):
        A0, B0 = J[a], J[b]
        d = (B0 - A0) / np.linalg.norm(B0 - A0)
        t0 = (X0[on] - J[bm]) @ d
        for lo in np.arange(-0.04, 0.04, 0.01):
            s = (t0 >= lo) & (t0 < lo + 0.01) & ((t0 < 0) if seg == 0 else (t0 >= 0))
            if s.sum() < 15:
                continue
            V0, V1 = X0[on][s], X1[on][s]
            r0 = np.linalg.norm((V0 - V0.mean(0)) - np.outer((V0 - V0.mean(0)) @ d, d), axis=1).mean()
            # bent: about the slice's own centre and its segment's bent direction
            d1 = d if seg == 0 else (J1[b] - J1[a]) / np.linalg.norm(J1[b] - J1[a])
            r1 = np.linalg.norm((V1 - V1.mean(0)) - np.outer((V1 - V1.mean(0)) @ d1, d1), axis=1).mean()
            ratios.append(r1 / max(r0, 1e-9))
    return over, float(min(ratios)) if ratios else float("nan")


res, tiles = {}, []
poses = {"rest": np.concatenate([np.eye(3), np.zeros((3, 1))], 1)[None].repeat(25, 0)}
for fr in (500, 3000, 6000, 9000):
    poses["recorded %d" % fr] = bones[fr]
for name, T in poses.items():
    X0 = lbs(P, idx, sw, T)
    row = [render(X0, F, UV0, grey, "side", size=300)]
    for deg in (36, 45):
        Tb = T.copy()
        for p in range(4):
            Tb_p, J = bent(T, p, deg)
            X1 = lbs(P, idx, sw, Tb_p)
            over, girth = measure(X0, X1, p, J, joints_of(Tb_p[None], Jr)[0])
            res.setdefault(name, {})["%s %d deg" % (PAWS[p], deg)] = {"edges_over_30pct": over, "girth_min_ratio": girth}
            Tb, _ = bent(Tb, p, deg)
        row.append(render(lbs(P, idx, sw, Tb), F, UV0, grey, "side", size=300))
    tiles.append(np.concatenate(row, 1))
Image.fromarray((np.concatenate(tiles, 0) * 255).clip(0, 255).astype(np.uint8)).save(os.path.join(out, "bend_side.png"))
# close-ups of each leg in the rest pose: unbent | 36 | 45, side view cropped to the leg
T = poses["rest"]
rows = []
for p in range(4):
    on = leg_of == p
    crop = np.zeros(len(P), bool)
    cen = P[on].mean(0)
    crop = np.linalg.norm(P - cen, axis=1) < 0.12
    row = []
    for deg in (0, 36, 45):
        Tb, _ = bent(T, p, deg) if deg else (T, None)
        X = lbs(P, idx, sw, Tb)
        view = "side" if PAWS[p].startswith("+x") else "other_side"
        row.append(render(X, F, UV0, grey, view, size=300, crop=crop))
    rows.append(np.concatenate(row, 1))
Image.fromarray((np.concatenate(rows, 0) * 255).clip(0, 255).astype(np.uint8)).save(os.path.join(out, "bend_legs_rest.png"))
summary = {}
for deg in (36, 45):
    ov = [v["edges_over_30pct"] for n in res.values() for k, v in n.items() if k.endswith("%d deg" % deg)]
    gi = [v["girth_min_ratio"] for n in res.values() for k, v in n.items() if k.endswith("%d deg" % deg)]
    summary["%d deg" % deg] = {"edges_over_30pct_worst": max(ov), "edges_over_30pct_median": float(np.median(ov)),
                               "girth_min_ratio_worst": float(np.nanmin(gi)), "girth_min_ratio_median": float(np.nanmedian(gi)),
                               "passes_numbers": bool(max(ov) <= 0.10 and np.nanmin(gi) >= 0.80)}
json.dump({"summary": summary, "per_pose": res}, open(os.path.join(out, "bend.json"), "w"), indent=1)
print(json.dumps(summary, indent=1))
for name, d in res.items():
    print(name, {k: (round(v["edges_over_30pct"], 3), round(v["girth_min_ratio"], 2)) for k, v in d.items() if k.endswith("36 deg")})
