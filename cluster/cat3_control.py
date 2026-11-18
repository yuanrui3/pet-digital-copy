"""The whole Body tab cat at home, one-day test, check 1 (work/pets/PROTOCOL_CAT.md): the Body tab (C1) cat moved by
its own bones with plain skinning (what the page can do) against its own exported frames (Lab4D's full warp, as the
Body tab shows them). Each frame is lined up by one scale, rotation and shift first, so only the shape counts.
CPU only.
usage: python cat3_control.py TOOLS_DIR PAGE_DIR UV_NPZ SKEL_NPZ C1_TEX OUT_DIR
  PAGE_DIR: cat.bin/.json (the Body tab's 92 posed frames); UV_NPZ: c1_uv.npz; SKEL_NPZ: home2_skel.py's output"""
import json
import os
import sys
sys.path.insert(0, sys.argv[1])
import numpy as np
from PIL import Image
from home_coat import load_page, render
from home2_transfer import lbs, umeyama

page, uv_npz, skel_npz, c1_tex, out = sys.argv[2:7]
os.makedirs(out, exist_ok=True)
m, C = load_page(page, "cat")
lo, hi = np.array(m["lo"]), np.array(m["hi"])
frames = lo + (hi - lo) * C["pos"].astype(np.float64) / 65535.0  # (92, V, 3)
z = np.load(uv_npz)
Q, F, UV = z["pos"].astype(np.float64), z["faces"].astype(np.int64), z["uv"].astype(np.float64)
sk = np.load(skel_npz)
own, own_raw = sk["own"].astype(np.float64), sk["own_frames"].astype(int)
i4, w4 = sk["skin_idx"].astype(np.int64), sk["skin_w"].astype(np.float64)
assert frames.shape[1] == len(Q), (frames.shape, Q.shape)
raw = m["raw_frames"]
pairs = [(f, int(np.where(own_raw == r)[0][0])) for f, r in enumerate(raw) if r in set(own_raw.tolist())]
gaps, rows = [], []
for n, (f, k) in enumerate(pairs):
    Y = lbs(Q, i4, w4, own[k])
    s, R, t = umeyama(Y, frames[f])
    Ya = s * Y @ R.T + t
    d = np.linalg.norm(Ya - frames[f], axis=1)
    gaps.append([np.median(d), np.percentile(d, 95), d.max()])
    if n in (0, len(pairs) // 3, 2 * len(pairs) // 3, len(pairs) - 1):
        T = np.asarray(Image.open(c1_tex).convert("RGB"), dtype=np.float32) / 255.0
        rows.append(np.concatenate([render(frames[f], F, UV, T, "side", size=420), render(Ya, F, UV, T, "side", size=420),
                                    render(frames[f], F, UV, T, "front", size=420), render(Ya, F, UV, T, "front", size=420)], 1))
gaps = np.array(gaps)
res = {"frames_compared": len(pairs), "median_gap_mm_median_over_frames": 1000 * float(np.median(gaps[:, 0])),
       "median_gap_mm_worst_frame": 1000 * float(gaps[:, 0].max()), "p95_gap_mm_median_over_frames": 1000 * float(np.median(gaps[:, 1])),
       "max_gap_mm_worst_frame": 1000 * float(gaps[:, 2].max()), "skin_top4_cover": float(sk["skin_cover"])}
json.dump(res, open(os.path.join(out, "control.json"), "w"), indent=1)
print(json.dumps(res, indent=1))
Image.fromarray((np.concatenate(rows, 0) * 255).clip(0, 255).astype(np.uint8)).save(os.path.join(out, "control.png"))
print("columns: exported side | own bones side | exported front | own bones front", flush=True)
