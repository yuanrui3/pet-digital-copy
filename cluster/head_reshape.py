"""The head at home given the C1 cat's head shape: the home mesh bent onto C1 (home_coat.py SAVE_BENT), its head moved
back rigidly onto the home head (so it stays where the skeleton holds it), the crinkles of the bend smoothed out of the
change, and blended into the neck. CPU, exploration.
usage: python head_reshape.py TOOLS_DIR PAGE_DIR BENT_NPZ TEX OUT_DIR [SMOOTH_ITERS]
Writes OUT_DIR/head.npz (pos: new rest positions) and renders: rest pose (current | new) and posed frames."""
import os
import sys
sys.path.insert(0, sys.argv[1])
import numpy as np
from PIL import Image
from scipy.sparse import coo_matrix, diags
from home_coat import load_page, render
from home2_stand import weld
from home2_transfer import lbs
page, bent, tex_path, out = sys.argv[2:6]
iters = int(sys.argv[6]) if len(sys.argv) > 6 else 20
FULL_BACK, ZERO_BACK = float(os.environ.get("FULL_BACK", 0.08)), float(os.environ.get("ZERO_BACK", 0.12))
FULL_BELOW, ZERO_BELOW = 0.05, 0.09
os.makedirs(out, exist_ok=True)
am, A = load_page(page, "a2s_cat")
F, UV = A["faces"].astype(np.int64), A["uv"].astype(np.float64)
z = np.load(bent)
P, Pb, nose = z["P"], z["Pb"], z["nose"]
assert np.allclose(P, A["pos"].astype(np.float64))
T = np.asarray(Image.open(tex_path).convert("RGB"), dtype=np.float32) / 255.0
sm = lambda x: x * x * (3 - 2 * x)  # noqa: E731
back, below = P[:, 2] - P[:, 2].min(), P[:, 1] - nose[1]
w = sm(np.clip((ZERO_BACK - back) / (ZERO_BACK - FULL_BACK), 0, 1)) * sm(np.clip((ZERO_BELOW - below) / (ZERO_BELOW - FULL_BELOW), 0, 1))
full = w > 0.99
# rigid: the bent head onto the home head (Kabsch on the fully weighted part)
mp, mb = P[full].mean(0), Pb[full].mean(0)
U, s, Vt = np.linalg.svd((Pb[full] - mb).T @ (P[full] - mp))
R = (U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt).T
Pa = (Pb - mb) @ R.T + mp
ang = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
print("rigid: turn %.1f deg, shift %.1f mm; head vertices %d full, %d blended" % (ang, 1000 * np.linalg.norm(mp - mb), full.sum(), ((w > 0.01) & ~full).sum()))
# the change, on the welded mesh (seams stay closed), smoothed by repeated neighbour averaging
wid, Pw, e = weld(P, F)
n = len(Pw)
Dw = np.zeros((n, 3)); cnt = np.zeros(n)
np.add.at(Dw, wid, Pa - P); np.add.at(cnt, wid, 1); Dw /= cnt[:, None]
ww = np.zeros(n); np.add.at(ww, wid, w); ww /= cnt
Adj = coo_matrix((np.ones(2 * len(e)), (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])), shape=(n, n)).tocsr()
Mavg = diags(1 / np.maximum(np.asarray(Adj.sum(1)).ravel(), 1)) @ Adj
Ds = Dw.copy()
for _ in range(iters):
    Ds = 0.5 * Ds + 0.5 * (Mavg @ Ds)
Pw_new = Pw + ww[:, None] * Ds
Pn = Pw_new[wid]
d = np.linalg.norm(Pn - P, axis=1)
rough = np.linalg.norm(Dw - Mavg @ Dw, axis=1)[ww > 0.99]
rough_s = np.linalg.norm(Ds - Mavg @ Ds, axis=1)[ww > 0.99]
print("smoothing %d: head change median %.1f mm, max %.1f; crinkle (change minus neighbours' mean) median %.2f -> %.2f mm, p99 %.2f -> %.2f" % (
    iters, 1000 * np.median(d[full]), 1000 * d.max(), 1000 * np.median(rough), 1000 * np.median(rough_s), 1000 * np.percentile(rough, 99), 1000 * np.percentile(rough_s, 99)))
np.savez(os.path.join(out, "head.npz"), pos=Pn.astype(np.float32), w=w)
# the page's file: vertex numbers and rest-pose offsets of the vertices that move (a2s_head.bin/.json, as viewer/data)
mv = np.where(np.linalg.norm(Pn - P, axis=1) > 1e-6)[0].astype(np.uint32)
off = (Pn - P)[mv].astype(np.float32)
blob = mv.tobytes() + off.tobytes()
open(os.path.join(out, "a2s_head.bin"), "wb").write(blob)
import json
json.dump({"blobs": [{"name": "idx", "dtype": "uint32", "shape": [len(mv)], "offset": 0, "bytes": int(mv.nbytes)},
                     {"name": "off", "dtype": "float32", "shape": [len(mv), 3], "offset": int(mv.nbytes), "bytes": int(off.nbytes)}],
           "bytes": len(blob), "units": "metres, the cat's own (root) frame, added to a2s_cat pos",
           "from": "head_reshape.py: the home mesh bent onto the C1 cat (smoothed), its head moved back rigidly onto the home head, blended into the neck",
           "chunks": 1}, open(os.path.join(out, "a2s_head.json"), "w"))
print("page file: %d vertices move" % len(mv), flush=True)
crop = (P[:, 2] < P[:, 2].min() + 0.12) & (P[:, 1] < nose[1] + 0.06)
for view in ("front", "side", "other_side", "three_quarter"):
    row = [render(X, F, UV, T, view, crop=crop) for X in (P, Pn)]
    Image.fromarray((np.concatenate(row, 1) * 255).clip(0, 255).astype(np.uint8)).save(os.path.join(out, "rest_%s.png" % view))
row = [render(X, F, UV, T, "side") for X in (P, Pn)]
Image.fromarray((np.concatenate(row, 1) * 255).clip(0, 255).astype(np.uint8)).save(os.path.join(out, "rest_side_whole.png"))
# posed: a few recorded frames, current | new, side and front of the head
bones = A["bones"].astype(np.float64).reshape(len(A["bones"]), 25, 3, 4)
idx, sw = A["skin_idx"].astype(np.int64), A["skin_w"].astype(np.float64)
for fi in (500, 3000, 6000, 9000):
    Xo, Xn = lbs(P, idx, sw, bones[fi]), lbs(Pn, idx, sw, bones[fi])
    hc = np.linalg.norm(Xo - Xo[np.argsort(P[:, 2])[:20]].mean(0), axis=1) < 0.13
    row = [render(X, F, UV, T, v, crop=hc) for v in ("side", "front") for X in (Xo, Xn)]
    Image.fromarray((np.concatenate(row, 1) * 255).clip(0, 255).astype(np.uint8)).save(os.path.join(out, "posed_%05d.png" % fi))
print("renders done", flush=True)
