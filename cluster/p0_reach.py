"""Phase 0, P0-3 (work/pets/PROTOCOL_CAT.md, "Restart, Phase 0"): how far must the hips or shoulders drop so that a
held paw is reached at 95% of the leg's length, instead of lengthening the leg as version 18 does?

For every frame and paw that version 18's planted-paw solver holds (its correction is non-zero and the paw is down),
in the cat's own frame and before any bending: top joint H, wrist or hock W, held wrist target W + correction. The
smallest lowering s along the room's "down" with |W + corr - (H + s down)| <= 0.95 (a + b) (a, b: the chain's two
segments) is the drop that leg needs; a girdle (hips: both hind legs, shoulders: both front legs) must drop by the larger
of its two. Reported as a share of leg length, with how fast it changes between frames. CPU only.
usage: python p0_reach.py PAGE_DIR CHECK_DIR IK_DIR OUT_JSON
  PAGE_DIR: feet_check.py's page folder; CHECK_DIR: its feet_frames.npz; IK_DIR: feet_ik.py's output (feet_ik.bin/.json)"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feet_check import CHUNK, DOWN, Page  # noqa: E402
from feet_ik import CHAINS, joints_of  # noqa: E402
from home2_transfer import read_blobs  # noqa: E402

page_dir, check_dir, ik_dir, out = sys.argv[1:5]
pg = Page(page_dir)
F = np.load(os.path.join(check_dir, "feet_frames.npz"))
im = json.load(open(os.path.join(ik_dir, "feet_ik.json")))
IK = read_blobs(os.path.join(ik_dir, "feet_ik.bin"), im)
off = np.concatenate([IK["rec"], IK["walk"]]).astype(np.float64)  # (n, 4, 3) cat's own frame
Jr = np.array(im["rest_joints"], dtype=np.float64)
T = np.concatenate([tr["bones"] for tr in pg.tracks]).astype(np.float64)
Hroom = np.concatenate([tr["H"] for tr in pg.tracks]).astype(np.float64)
kind, track = F["kind"], F["track"]
assert len(T) == len(off) == len(kind), (len(T), len(off), len(kind))
down_cat = Hroom[:, 1, :]  # the room's down (y) in the cat's own frame: H^T e_y = the second row of H
held = (np.abs(off).max(2) > 1e-3) & (F["hP"] <= DOWN)
need = np.full((len(T), 4), np.nan)  # drop needed per leg, metres (nan: not held; inf: no dip reaches it)
L = np.zeros((len(T), 4))
for a0 in range(0, len(T), CHUNK):
    a1 = min(len(T), a0 + CHUNK)
    J = joints_of(T[a0:a1], Jr)
    for p, (bt, bm, bw, bp) in enumerate(CHAINS):
        Hp, Kp, Wp = J[:, bt], J[:, bm], J[:, bw]
        a, b = np.linalg.norm(Kp - Hp, axis=1), np.linalg.norm(Wp - Kp, axis=1)
        R = 0.95 * (a + b)
        L[a0:a1, p] = a + b
        dv = Wp + off[a0:a1, p] - Hp
        c = np.einsum("ni,ni->n", dv, down_cat[a0:a1])
        d2 = np.einsum("ni,ni->n", dv, dv)
        disc = c * c - d2 + R * R
        s = np.where(d2 <= R * R, 0.0, np.where((disc >= 0) & (c > 0), c - np.sqrt(np.maximum(disc, 0)), np.inf))
        need[a0:a1, p] = np.where(held[a0:a1, p], np.maximum(s, 0), np.nan)
res = {}
girdles = {"shoulders (front legs)": [0, 1], "hips (hind legs)": [2, 3]}
for k in ("rec", "walk"):
    mk = kind == k
    for g, legs in girdles.items():
        n = need[mk][:, legs]
        Lg = L[mk][:, legs].mean(1)
        anyheld = ~np.isnan(n).all(1)
        gd = np.nanmax(np.where(np.isnan(n), -1, n), axis=1)[anyheld]  # the girdle drops by the larger need
        frac = gd / Lg[anyheld]
        fin = np.isfinite(frac)
        # change between consecutive frames of the same track where the girdle holds a paw in both
        idx = np.where(anyheld)[0]
        tr = track[mk][idx]
        same = (np.diff(idx) == 1) & (tr[1:] == tr[:-1]) & fin[1:] & fin[:-1]
        dd = np.abs(np.diff(gd))[same]
        res["%s %s" % (k, g)] = {
            "frames_with_a_held_paw": int(anyheld.sum()),
            "no_dip_reaches_share": float((~fin).mean()),
            "drop_share_of_leg_p50_p90_p95": [float(np.percentile(frac[fin], q)) for q in (50, 90, 95)],
            "share_within_10pct_of_leg": float((frac[fin] <= 0.10).sum() / len(frac)),
            "share_needing_no_drop": float((frac[fin] == 0).sum() / len(frac)),
            "drop_cm_p50_p95": [100 * float(np.percentile(gd[fin], q)) for q in (50, 95)],
            "change_cm_per_0.1s_p50_p95": [100 * float(np.percentile(dd, q)) for q in (50, 95)] if len(dd) else None,
            "share_changes_within_1cm": float((dd <= 0.01).mean()) if len(dd) else None,
            "leg_length_m": float(np.median(Lg)),
        }
        r = res["%s %s" % (k, g)]
        r["go_ahead"] = bool(r["share_within_10pct_of_leg"] >= 0.95 and (r["share_changes_within_1cm"] or 0) >= 0.95)
json.dump(res, open(out, "w"), indent=1)
print(json.dumps(res, indent=1))
