"""Smoothing the skin weights (work/pets/PROTOCOL_CAT.md, "Smoothing the skin weights"): heat diffusion of the body's
25 bone weights over its surface, on the legs and the skin around them, so neighbouring vertices no longer jump from
one bone to another.
  - one implicit step (M + t C) W' = M W: C the cotangent Laplacian (stiffness), M the lumped vertex areas, t the
    diffusion time (m^2; the spread is about sqrt(2 t));
  - smoothed: vertices with any weight (>= 0.01) on a leg bone, widened by RING rings of neighbours; held fixed:
    everything else, and the neck vertices the head graft reshapes (a2s_graft body_idx) with HOLD rings around them;
  - each vertex then keeps its four largest weights, renormalised (the page skins with four);
  - copies of a vertex at the same place (the coat's texture seams) are welded into one point first, so they keep
    the same weights and the seams stay shut.
Writes OUT_DIR/page: a copy of the page data folder (links) with a2s_cat.bin/.json holding the new weights, and
skin_smooth.json (what changed). CPU only.
usage: python skin_smooth.py PAGE_DIR GRAFT_DIR OUT_DIR   (env T, RING, HOLD)"""
import json
import os
import shutil
import sys

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import factorized

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feet_ik import CHAINS  # noqa: E402
from home2_transfer import pack, read_blobs  # noqa: E402

page_dir, graft_dir, out = sys.argv[1:4]
T = float(os.environ.get("T", 1e-4))
RING = int(os.environ.get("RING", 4))
HOLD = int(os.environ.get("HOLD", 2))
am = json.load(open(os.path.join(page_dir, "a2s_cat.json")))
A = read_blobs(os.path.join(page_dir, "a2s_cat.bin"), am)
gm = json.load(open(os.path.join(graft_dir, "a2s_graft.json")))
G = read_blobs(os.path.join(graft_dir, "a2s_graft.bin"), gm)
X0, F0 = A["pos"].astype(np.float64), A["faces"].astype(np.int64)
idx0, w0 = A["skin_idx"].astype(np.int64), A["skin_w"].astype(np.float64)
V0, B = len(X0), am["bones"]
W0 = np.zeros((V0, B))
np.add.at(W0, (np.repeat(np.arange(V0), 4), idx0.reshape(-1)), w0.reshape(-1))
# weld: the coat's texture seams carry copies of a vertex at the same place; they are one point here and get the same
# weights (smoothing them apart opens the seams)
_, rep, inv = np.unique(np.round(X0 / 1e-6).astype(np.int64), axis=0, return_index=True, return_inverse=True)
inv = inv.reshape(-1)
X, V = X0[rep], len(rep)
F = inv[F0]
F = F[(F[:, 0] != F[:, 1]) & (F[:, 1] != F[:, 2]) & (F[:, 2] != F[:, 0])]
W = W0[rep]
copies_weight_diff = float(np.abs(W0 - W[inv]).max())
idx, w = idx0[rep], w0[rep]

# cotangent stiffness C and lumped areas M
C = sp.csr_matrix((V, V))
rows, cols, vals = [], [], []
area = np.zeros(V)
for k in range(3):
    i, j, o = F[:, k], F[:, (k + 1) % 3], F[:, (k + 2) % 3]
    u, v = X[i] - X[o], X[j] - X[o]
    cr = np.linalg.norm(np.cross(u, v), axis=1)
    cot = (u * v).sum(1) / np.maximum(cr, 1e-12)
    rows += [i, j, i, j]
    cols += [j, i, i, j]
    vals += [-0.5 * cot, -0.5 * cot, 0.5 * cot, 0.5 * cot]
fa = 0.5 * np.linalg.norm(np.cross(X[F[:, 1]] - X[F[:, 0]], X[F[:, 2]] - X[F[:, 0]]), axis=1)
for k in range(3):
    np.add.at(area, F[:, k], fa / 3)
C = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(V, V))
adj = sp.csr_matrix((np.ones(6 * len(F)), (np.r_[F[:, 0], F[:, 1], F[:, 2], F[:, 1], F[:, 2], F[:, 0]],
                                           np.r_[F[:, 1], F[:, 2], F[:, 0], F[:, 0], F[:, 1], F[:, 2]])), shape=(V, V))


def grow(mask, n):
    for _ in range(n):
        mask = mask | (adj @ mask.astype(float) > 0)
    return mask


leg = W[:, CHAINS.reshape(-1)].sum(1) >= 0.01
region = grow(leg, RING)
neck = np.zeros(V, bool)
neck[inv[G["body_idx"].astype(np.int64)]] = True
hold = grow(neck, HOLD)
free = region & ~hold
fi, ci = np.where(free)[0], np.where(~free)[0]
M = sp.diags(area)
Aff = (M + T * C)[fi][:, fi].tocsc()
rhs = (M @ W)[fi] - (T * C)[fi][:, ci] @ W[ci]
solve = factorized(Aff)
W2 = W.copy()
W2[fi] = np.stack([solve(rhs[:, b]) for b in range(B)], 1)
W2 = np.clip(W2, 0, None)
top = np.argsort(-W2, axis=1)[:, :4]
tw = np.take_along_axis(W2, top, 1)
tw /= np.maximum(tw.sum(1, keepdims=True), 1e-12)
# held vertices keep exactly their old four
top[~free], tw[~free] = idx[~free], w[~free]
# what changed
Wn = np.zeros((V, B))
np.add.at(Wn, (np.repeat(np.arange(V), 4), top.reshape(-1)), tw.reshape(-1))
dW = np.abs(Wn - W).sum(1) / 2
E = np.unique(np.sort(np.r_[F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]], 1), axis=0)
jump0 = np.abs(W[E[:, 0]] - W[E[:, 1]]).sum(1) / 2
jump1 = np.abs(Wn[E[:, 0]] - Wn[E[:, 1]]).sum(1) / 2
inr = free[E[:, 0]] & free[E[:, 1]]
res = {"T_m2": T, "RING": RING, "HOLD": HOLD, "vertices": V0, "welded_points": V, "copies_weight_diff_before": copies_weight_diff,
       "smoothed_points": int(free.sum()), "held_neck": int((region & hold).sum()),
       "weight_change_q50_q90_max": np.percentile(dW[free], [50, 90, 100]).round(3).tolist(),
       "edge_weight_jump_in_region_before_q90_q99_max": np.percentile(jump0[inr], [90, 99, 100]).round(3).tolist(),
       "edge_weight_jump_in_region_after_q90_q99_max": np.percentile(jump1[inr], [90, 99, 100]).round(3).tolist(),
       "kept_weight_in_top4_min": float(np.take_along_axis(W2, np.argsort(-W2, axis=1)[:, :4], 1).sum(1)[free].min() / max(W2[free].sum(1).max(), 1e-12))}
print(json.dumps(res, indent=1))

# the new page data folder: links to everything, a2s_cat rewritten
pd = os.path.join(out, "page")
os.makedirs(pd, exist_ok=True)
for f in os.listdir(page_dir):
    if not f.startswith("a2s_cat."):
        dst = os.path.join(pd, f)
        if not os.path.lexists(dst):
            os.symlink(os.path.realpath(os.path.join(page_dir, f)), dst)
blobs = [(b["name"], A[b["name"]]) for b in am["blobs"]]
blobs = [(n, top[inv].astype(np.uint8) if n == "skin_idx" else tw[inv].astype(np.float32) if n == "skin_w" else a) for n, a in blobs]
meta, nbytes = pack(os.path.join(pd, "a2s_cat.bin"), blobs)
m2 = dict(am)
m2["blobs"], m2["bytes"] = meta, nbytes
m2["skin"] = "weights smoothed on the legs and around them, seams welded (skin_smooth.py: T %g m^2, RING %d, neck held)" % (T, RING)
json.dump(m2, open(os.path.join(pd, "a2s_cat.json"), "w"), indent=1)
json.dump(res, open(os.path.join(out, "skin_smooth.json"), "w"), indent=1)
