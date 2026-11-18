"""The head: the home mesh as it is (P), bent onto the C1 cat (Pb), and the C1 cat itself (Xc), all in the home rest frame.
usage: python head_diag.py TOOLS_DIR PAGE_DIR BENT_NPZ HOME_TEX C1_TEX OUT_DIR"""
import os
import sys
sys.path.insert(0, sys.argv[1])
import numpy as np
from PIL import Image
from home_coat import load_page, render
page, bent, home_tex, c1_tex, out = sys.argv[2:7]
os.makedirs(out, exist_ok=True)
am, A = load_page(page, "a2s_cat")
F, UV = A["faces"].astype(np.int64), A["uv"].astype(np.float64)
z = np.load(bent)
P, Pb, Xc, Fc, UVc, nose = z["P"], z["Pb"], z["Xc"], z["Fc"], z["UVc"], z["nose"]
T = np.asarray(Image.open(home_tex).convert("RGB"), dtype=np.float32) / 255.0
Tc = np.asarray(Image.open(c1_tex).convert("RGB"), dtype=np.float32) / 255.0
zmin = P[:, 2].min()


def head_mask(X):
    return (X[:, 2] < zmin + 0.12) & (X[:, 1] < nose[1] + 0.06)


def stats(name, X):
    h = X[(X[:, 2] < zmin + 0.08)]  # the front 8 cm
    top = h[:, 1].min()
    print("%-10s front 8 cm: width %.1f cm, height %.1f cm (top %.1f cm above the nose); nose at z %.1f cm; centre x %.1f cm" % (
        name, 100 * np.ptp(h[:, 0]), 100 * np.ptp(h[:, 1]), 100 * (nose[1] - top), 100 * X[:, 2].min(), 100 * h[:, 0].mean()))


stats("home", P); stats("bent", Pb); stats("C1", Xc)
hm = head_mask(P)
d = np.linalg.norm(Pb - P, axis=1)
print("bent - home displacement on the head: median %.1f mm, p90 %.1f, max %.1f; whole body median %.1f mm" % (
    1000 * np.median(d[hm]), 1000 * np.percentile(d[hm], 90), 1000 * d[hm].max(), 1000 * np.median(d)))
# rigid part of the head change (Kabsch, home head -> bent head)
a, b = P[hm] - P[hm].mean(0), Pb[hm] - Pb[hm].mean(0)
U, s, Vt = np.linalg.svd(a.T @ b)
R = (U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt).T
ang = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
print("head: rigid turn home -> bent %.1f deg, shift %.1f mm; after removing it, shape change median %.1f mm" % (
    ang, 1000 * np.linalg.norm(Pb[hm].mean(0) - P[hm].mean(0)), 1000 * np.median(np.linalg.norm(b - a @ R.T, axis=1))))
for view in ("side", "front", "top", "three_quarter", "other_side"):
    crop = None if view == "top" else hm
    row = [render(P, F, UV, T, view, crop=crop), render(Pb, F, UV, T, view, crop=head_mask(Pb) if crop is not None else None),
           render(Xc, Fc, UVc, Tc, view, crop=head_mask(Xc) if crop is not None else None)]
    Image.fromarray((np.concatenate(row, 1) * 255).clip(0, 255).astype(np.uint8)).save(os.path.join(out, "head_%s.png" % view))
    print("render %s done" % view, flush=True)
