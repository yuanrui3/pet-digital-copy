"""Where the walking cycle stretches the skin (work/pets/PROTOCOL_CAT.md, walking cycle, before round 2): on gait_check's
200 frames, the edges stretched over 30%, grouped by the bone that mostly moves each edge, for version 22 and the cycle.
usage: python gait_stretch_where.py PAGE_DIR IK_DIR GAIT_DIR"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feet_check import Page  # noqa: E402
from feet_ik import CHAINS, bend  # noqa: E402
from home2_transfer import edges_of, lbs, read_blobs, stretch  # noqa: E402

page_dir, ik_dir, gait_dir = sys.argv[1:4]
pg = Page(page_dir)
im = json.load(open(os.path.join(ik_dir, "feet_ik.json")))
IKW = read_blobs(os.path.join(ik_dir, "feet_ik.bin"), im)["walk"].astype(np.float64)
Jr = np.array(im["rest_joints"], dtype=np.float64)
GB = np.load(os.path.join(gait_dir, "gait_bones.npz"))
walks = [tr for tr in pg.tracks if tr["kind"] == "walk"]
rows, base = [], 0
for tr in walks:
    rows.append(base)
    base += len(tr["bones"])
e = edges_of(pg.fp)
main = pg.ip[np.arange(len(pg.P)), np.argmax(pg.wp, 1)]
eb = main[e[:, 0]]
legb = {int(b): name for name, ch in zip(["front -x", "front +x", "hind -x", "hind +x"], CHAINS) for b in ch}
group = np.array([legb.get(int(b), "body, head, tail") for b in eb])
rng = np.random.default_rng(1)
tot = {}
for _ in range(200):
    ti = rng.integers(len(walks))
    n = len(walks[ti]["bones"])
    k = int(rng.integers(n))
    f = rows[ti] + k
    T = walks[ti]["bones"][k:k + 1].astype(np.float64)
    for name, Tb in (("version 22", bend(T, Jr, IKW[f:f + 1])[0]), ("walking cycle", GB["bones"][f].astype(np.float64))):
        s = stretch(pg.P, lbs(pg.P, pg.ip, pg.wp, Tb), e) > np.log(1.3)
        for g in np.unique(group):
            tot.setdefault(name, {}).setdefault(g, 0)
            tot[name][g] += int(s[group == g].sum())
nE = len(e) * 200
for g in sorted(set(group)):
    print("%-18s edges %5d  over 30%%: version 22 %.2f%%  cycle %.2f%%  (points of all edges: +%.2f)" % (
        g, (group == g).sum(), 100 * tot["version 22"][g] / ((group == g).sum() * 200), 100 * tot["walking cycle"][g] / ((group == g).sum() * 200),
        100 * (tot["walking cycle"][g] - tot["version 22"][g]) / nE))
