"""Which part of the cat is lowest on the copy's walks (the walking cycle's ground). CPU only.
usage: python gait_lowest.py PAGE_DIR"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feet_check import Page, skin  # noqa: E402

pg = Page(sys.argv[1])
walks = [tr for tr in pg.tracks if tr["kind"] == "walk"]
sel = pg.e3p
main_bone = pg.ip[sel, np.argmax(pg.wp[sel], 1)]
paws = np.zeros(len(pg.P), bool)
for s in pg.lp[:4]:
    paws[s] = True
cnt, gap = np.zeros(25), []
for tr in walks[::4]:
    T, H = tr["bones"][::5].astype(np.float64), tr["H"][::5]
    X = skin(T, pg.P[sel], pg.ip[sel], pg.wp[sel])
    h = np.einsum("nvi,ni->nv", X, H[:, 1, :])
    lo = h.argmax(1)
    np.add.at(cnt, main_bone[lo], 1)
    # lowest paw vertex against the lowest point
    hp = np.where(paws[sel][None], h, -np.inf).max(1)
    gap.append(h.max(1) - hp)
gap = np.concatenate(gap)
print("lowest point's main bone (share of frames): %s" % {int(b): round(c / cnt.sum(), 3) for b, c in enumerate(cnt) if c})
print("lowest paw vertex above the lowest point (m): q10 %.3f q50 %.3f q90 %.3f" % tuple(np.percentile(gap, [10, 50, 90])))
