"""The walking cycle (work/pets/PROTOCOL_CAT.md, "A built-in walking cycle on the copy's walks"): facts the design
needs, read off the page's own data before anything is built. Per leg: segment lengths, how straight the leg is while
its paw is down, which way its middle joint bends (towards the head or the tail), and where the paw sits relative to
the top joint; per walk: how fast the shoulders and hips travel. CPU only.
usage: python gait_probe.py PAGE_DIR IK_DIR"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feet_check import Page  # noqa: E402
from feet_ik import CHAINS, joints_of  # noqa: E402

page_dir, ik_dir = sys.argv[1:3]
pg = Page(page_dir)
Jr = np.array(json.load(open(os.path.join(ik_dir, "feet_ik.json")))["rest_joints"], dtype=np.float64)
NAMES = ["front -x", "front +x", "hind -x", "hind +x"]
print("rest joints: head end z %.3f, tail end z %.3f (y down)" % (Jr[3, 2], Jr[16, 2]))
walks = [tr for tr in pg.tracks if tr["kind"] == "walk"]
T = np.concatenate([tr["bones"] for tr in walks]).astype(np.float64)
J = joints_of(T, Jr)
for p, (bt, bm, bw, bp) in enumerate(CHAINS):
    a = np.linalg.norm(J[:, bm] - J[:, bt], axis=-1)
    b = np.linalg.norm(J[:, bw] - J[:, bm], axis=-1)
    c = np.linalg.norm(J[:, bp] - J[:, bw], axis=-1)
    d = np.linalg.norm(J[:, bw] - J[:, bt], axis=-1)
    ext = d / (a + b)
    # bend direction: the middle joint's offset from the top-to-wrist line, along the body's z (towards the tail)
    u = (J[:, bw] - J[:, bt]) / d[:, None]
    off = (J[:, bm] - J[:, bt]) - ((J[:, bm] - J[:, bt]) * u).sum(-1, keepdims=True) * u
    tail = np.sign(off[:, 2])
    hz = (J[:, bw] - J[:, bt])
    print("%-9s upper %.3f lower %.3f wrist-paw %.3f m | extension median %.4f (p5 %.4f) | middle joint towards the tail "
          "in %.0f%% of frames | wrist below top: y %.3f, z %+.3f, x %+.3f" % (
              NAMES[p], np.median(a), np.median(b), np.median(c), np.median(ext), np.percentile(ext, 5),
              100 * (tail > 0).mean(), np.median(hz[:, 1]), np.median(hz[:, 2]), np.median(hz[:, 0])))
# girdle speeds in the room (10 frames a second)
sp = []
for tr in walks:
    Jt = joints_of(tr["bones"].astype(np.float64), Jr)
    G = Jt[:, [CHAINS[0][0], CHAINS[1][0], CHAINS[2][0], CHAINS[3][0]]]
    W = np.einsum("nij,ngj->ngi", tr["H"], G) + tr["root"][:, None]
    v = np.linalg.norm(np.diff(W[:, :, [0, 2]], axis=0), axis=-1).mean(1) * 10
    sp.append(v)
sp = np.concatenate(sp)
print("girdle speed (m/s) over all walk frames: p25 %.3f p50 %.3f p75 %.3f p90 %.3f p99 %.3f; share above 0.05: %.2f" % tuple(
    list(np.percentile(sp, [25, 50, 75, 90, 99])) + [(sp > 0.05).mean()]))
H = np.concatenate([tr["H"] for tr in walks])
print("cat's down axis in the room (row 1 of H), median: %s" % np.round(np.median(H[:, 1, :], axis=0), 3))
# standing or not: per walk frame, how many paws are down (feet_check: tip within 2 cm of the cat's lowest point) and
# how high the shoulders and hips are above the lowest point, against their median
from feet_check import DOWN  # noqa: E402
nd, hr = [], []
for tr in walks:
    Tt = tr["bones"].astype(np.float64)
    XP, _, lowP = pg.pose(Tt, tr["H"])
    L, _ = pg.split(XP, pg.e3p, pg.lp)
    down = tr["H"][:, 1, :]
    h = np.einsum("npi,ni->np", L[:, :4], down)
    nd.append(((lowP[:, None] - h) <= DOWN).sum(1))
    Jt = joints_of(Tt, Jr)
    g = np.einsum("npi,ni->np", Jt[:, [CHAINS[0][0], CHAINS[1][0], CHAINS[2][0], CHAINS[3][0]]], down)
    hr.append((lowP[:, None] - g))
nd, hr = np.concatenate(nd), np.concatenate(hr)
med = np.median(hr, 0)
print("paws down per walk frame: 0..4 shares %s" % np.round(np.bincount(nd, minlength=5) / len(nd), 3))
print("girdle heights above the lowest point, median per leg (m): %s; frames with every girdle above 85%% of its median: %.2f;"
      " above 70%%: %.2f" % (np.round(med, 3), (hr >= 0.85 * med).all(1).mean(), (hr >= 0.7 * med).all(1).mean()))
