"""Phase 0, P0-6 (work/pets/PROTOCOL_CAT.md, "Restart, Phase 0"): the shared measuring tool, first use. Measures the cat's
paws as the page shows them (version 22: the recordings as estimated; the copy's walks with version 18's planted-paw
bending, from feet_ik.bin), at the thresholds the user's error ladder set (P0-5: slides seen from 4 to 8 mm per 0.1 s)
and the project's old line (15 mm). Also: leg extension while a paw is down. With a second IK_DIR, the same frames are
measured for that candidate too (paired). CPU only.
usage: python p0_measure.py PAGE_DIR CHECK_DIR IK_DIR OUT_JSON [IK_DIR_B]   (REC_TOO=1: the planted paws on the recordings too)"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feet_check import CHUNK, DOWN, Page  # noqa: E402
from feet_ik import CHAINS, bend, joints_of  # noqa: E402
from home2_transfer import read_blobs  # noqa: E402

page_dir, check_dir, ik_dir, out = sys.argv[1:5]
ik_b = sys.argv[5] if len(sys.argv) > 5 else None
pg = Page(page_dir)
F = np.load(os.path.join(check_dir, "feet_frames.npz"))
kind, track = F["kind"], F["track"]
T = np.concatenate([tr["bones"] for tr in pg.tracks]).astype(np.float64)
Hr = np.concatenate([tr["H"] for tr in pg.tracks])
Rr = np.concatenate([tr["root"] for tr in pg.tracks])
THRESH = (0.004, 0.008, 0.015)


def offsets(d):
    im = json.load(open(os.path.join(d, "feet_ik.json")))
    IK = read_blobs(os.path.join(d, "feet_ik.bin"), im)
    off = np.concatenate([IK["rec"], IK["walk"]]).astype(np.float64)
    if os.environ.get("REC_TOO") != "1":
        off[kind == "rec"] = 0  # the page bends legs on the copy's walks only (REC_TOO=1: on the recordings too)
    return off, np.array(im["rest_joints"], dtype=np.float64)


def measure(off, Jr):
    paw, low = np.zeros((len(T), 4, 3)), np.zeros(len(T))
    ext = np.zeros((len(T), 4))
    for a in range(0, len(T), CHUNK):
        b = min(len(T), a + CHUNK)
        T2 = bend(T[a:b], Jr, off[a:b]) if np.abs(off[a:b]).max() > 0 else T[a:b]
        J = joints_of(T2, Jr)
        for p, (bt, bm, bw, bp) in enumerate(CHAINS):
            full = np.linalg.norm(J[:, bm] - J[:, bt], axis=1) + np.linalg.norm(J[:, bw] - J[:, bm], axis=1)
            ext[a:b, p] = np.linalg.norm(J[:, bw] - J[:, bt], axis=1) / full
        XP, _, lowP = pg.pose(T2, Hr[a:b])
        LP, _ = pg.split(XP, pg.e3p, pg.lp)
        paw[a:b] = np.einsum("nij,nlj->nli", Hr[a:b], LP[:, :4]) + Rr[a:b, None]
        low[a:b] = lowP + Rr[a:b, 1]
    h = low[:, None] - paw[..., 1]
    down = h <= DOWN
    same = np.r_[track[1:] == track[:-1], False]
    step = down[:-1] & down[1:] & same[:-1, None]
    skid = np.linalg.norm((paw[1:] - paw[:-1])[..., [0, 2]], axis=-1)
    res = {}
    for k in ("rec", "walk"):
        m = step & (kind[:-1] == k)[:, None]
        s = skid[m]
        dk = down & (kind == k)[:, None]
        res[k] = {"down_paw_steps": int(m.sum()),
                  "share_sliding_over_mm": {str(int(1000 * t)): float((s > t).mean()) for t in THRESH},
                  "slide_mm_p50_p90": [1000 * float(np.percentile(s, q)) for q in (50, 90)],
                  "leg_extension_while_down_p50_p90": [float(np.percentile(ext[dk], q)) for q in (50, 90)]}
    return res, skid, step


offA, JrA = offsets(ik_dir)
resA, skidA, stepA = measure(offA, JrA)
out_d = {"A (" + ik_dir + ")": resA}
if ik_b:
    offB, JrB = offsets(ik_b)
    resB, skidB, stepB = measure(offB, JrB)
    out_d["B (" + ik_b + ")"] = resB
    both = stepA & stepB  # paired: the same paw-steps, down in both
    for k in ("rec", "walk"):
        m = both & (kind[:-1] == k)[:, None]
        out_d["paired " + k] = {"steps": int(m.sum()), **{"A_over_%dmm" % int(1000 * t): float((skidA[m] > t).mean()) for t in THRESH},
                                **{"B_over_%dmm" % int(1000 * t): float((skidB[m] > t).mean()) for t in THRESH}}
json.dump(out_d, open(out, "w"), indent=1)
print(json.dumps(out_d, indent=1))
