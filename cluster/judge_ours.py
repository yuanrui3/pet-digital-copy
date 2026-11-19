"""The judge without ratings (work/pets/PROTOCOL_CAT.md, "Building the judge"): our cat in the same form as the real
clips. The page's cat (version 22's shape: the home body fitted to the grafted head, and the head piece) is posed as the
page poses it, drawn as surface points from a still, level camera beside its path, and measured with the same pair
measures as the real clips (judge_feat.py), with the exact image motion of every point and the ground at rest.

  - windows: NPAIR (8) pairs; the pair spacing (1 to 4 samples of 0.1 s) is chosen per window so that the travel per pair is
    close to TARGET sizes (the real clips' typical travel per pair);
  - camera: orthographic, 2 mm per pixel, 55 to 125 degrees from the travel direction (either side), 5 to 30 degrees
    above the horizontal, no roll; drawn at random with SEED, and saved, so candidates are measured on the same windows
    from the same cameras (WINDOWS=file reuses a list);
  - BLUR: the motion field is smoothed by this share of the cat's size (default 0.03), the ground's motion (none)
    mixing in at the outline, as an estimator's would;
  - PLANT: which tracks get the planted-paw bending (feet_ik.bin): "walk" (version 22 as shown), "none", "both";
  - SLIDE_MM: on the walks' held paws, a drift forward of this many mm per 0.1 s (the error ladder's defect);
  - TRACKS: "rec", "walk" (the 16 free walks) or "both"; MAXWIN: at most this many windows, drawn at random;
  - GAIT: gait_bones.npz (gait_cycle.py), the walking cycle in place of the walks' bones and floor shift.
CPU only.
usage: python judge_ours.py PAGE_DIR IK_DIR FLOOR_DIR GRAFT_NPZ OUT_NPZ"""
import json
import os
import sys

import numpy as np
from scipy.ndimage import binary_closing, distance_transform_edt, gaussian_filter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feet_check import Page  # noqa: E402
from feet_ik import bend  # noqa: E402
from home2_transfer import lbs, read_blobs  # noqa: E402
from judge_feat import NAMES, pair_measures  # noqa: E402

page_dir, ik_dir, floor_dir, graft, out = sys.argv[1:6]
TARGET = float(os.environ.get("TARGET", 0.05))
BLUR = float(os.environ.get("BLUR", 0.03))
PLANT = os.environ.get("PLANT", "walk")
SLIDE = float(os.environ.get("SLIDE_MM", 0)) / 1000
TRACKS = os.environ.get("TRACKS", "both")
MAXWIN = int(os.environ.get("MAXWIN", 250))
SEED = int(os.environ.get("SEED", 1))
NPAIR, PX, NS = int(os.environ.get("NPAIR", 8)), 500.0, 200000  # pairs per window, pixels per metre, surface points
SIZE0 = 0.30  # the cat's size (m) from the side, for the first guess of speed in sizes
GAIT = os.environ.get("GAIT")  # gait_bones.npz (gait_cycle.py): the walking cycle's bones and floor shift on the walks

pg = Page(page_dir)
g = np.load(graft)
Ps, Hh, Fh = g["Ps"].astype(np.float64), g["H"].astype(np.float64), g["Fh"].astype(np.int64)
hidx, hw = g["hidx"].astype(np.int64), g["hw"].astype(np.float64)
F = np.concatenate([pg.fp, Fh + len(Ps)])
im = json.load(open(os.path.join(ik_dir, "feet_ik.json")))
IK = read_blobs(os.path.join(ik_dir, "feet_ik.bin"), im)
Jr = np.array(im["rest_joints"], dtype=np.float64)
fm = json.load(open(os.path.join(floor_dir, "feet_floor.json")))
FL = read_blobs(os.path.join(floor_dir, "feet_floor.bin"), fm)
rng = np.random.default_rng(SEED)

# surface points: faces drawn by area at rest, random barycentric weights
R0 = np.concatenate([Ps, Hh])
fa = np.linalg.norm(np.cross(R0[F[:, 1]] - R0[F[:, 0]], R0[F[:, 2]] - R0[F[:, 0]]), axis=1)
fi = rng.choice(len(F), NS, p=fa / fa.sum())
u, v = rng.random(NS), rng.random(NS)
flip = u + v > 1
u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
bc = np.stack([1 - u - v, u, v], 1)

# tracks, each with its place in the recordings' / walks' data (for the floor fix and the planted paws)
tracks, nrec, nwalk = [], 0, 0
for tr in pg.tracks:
    n = len(tr["bones"])
    if tr["kind"] == "rec":
        tracks.append(dict(tr, base=nrec, src="rec"))
        nrec += n
    else:
        tracks.append(dict(tr, base=nwalk, src="walk"))
        nwalk += n
use = [k for k, tr in enumerate(tracks) if (tr["kind"] == "rec" and TRACKS in ("rec", "both"))
       or (tr["kind"] == "walk" and tr.get("sub") == "free" and TRACKS in ("walk", "both"))]


def offsets(tr, a, b):
    """the planted-paw corrections for frames a..b of a track (cat's own frame), or None"""
    if not (PLANT == "both" or PLANT == tr["src"]):
        return None
    off = IK[tr["src"]][tr["base"] + a:tr["base"] + b].astype(np.float64).copy()
    if SLIDE > 0 and tr["src"] == "walk":  # held paws drift forward, from where each hold began (at most 1 s)
        full = np.abs(IK["walk"][tr["base"]:tr["base"] + len(tr["bones"])].astype(np.float64)).max(2) > 1e-6
        run = np.zeros(full.shape)
        for t in range(1, len(full)):
            run[t] = np.where(full[t] & full[t - 1], np.minimum(run[t - 1] + 1, 10), 0)
        off[..., 2] -= SLIDE * run[a:b] * full[a:b]
    return off


GB = np.load(GAIT) if GAIT else None


def world_points(tr, a, b):
    """the surface points in the room for frames a..b (b - a, NS, 3), posed as the page poses them"""
    T = tr["bones"][a:b].astype(np.float64)
    off = offsets(tr, a, b)
    if GB is not None and tr["src"] == "walk":
        T = GB["bones"][tr["base"] + a:tr["base"] + b].astype(np.float64)
    elif off is not None and np.abs(off).max() > 0:
        T = bend(T, Jr, off)
    dy = FL["rec_dy" if tr["src"] == "rec" else "walk_dy"][tr["base"] + a:tr["base"] + b].astype(np.float64)
    if GB is not None and tr["src"] == "walk":
        dy = GB["dy"][tr["base"] + a:tr["base"] + b].astype(np.float64)
    out_ = np.zeros((b - a, NS, 3))
    for k in range(b - a):
        X = np.concatenate([lbs(Ps, pg.ip, pg.wp, T[k]), lbs(Hh, hidx, hw, T[k])])
        S = (bc[:, :, None] * X[F[fi]]).sum(1)
        out_[k] = S @ tr["H"][a + k].T + tr["root"][a + k]
        out_[k, :, 1] += dy[k]
    return out_


def camera(travel, az, el):
    """rows: image right, image down, view direction; a level camera az degrees from the travel direction, el above"""
    t = np.array([travel[0], 0.0, travel[2]])
    t /= np.linalg.norm(t)
    ca, sa = np.cos(np.radians(az)), np.sin(np.radians(az))
    hd = np.array([ca * t[0] + sa * t[2], 0.0, -sa * t[0] + ca * t[2]])
    view = np.cos(np.radians(el)) * hd + np.sin(np.radians(el)) * np.array([0.0, 1.0, 0.0])
    right = np.cross([0.0, 1.0, 0.0], view)
    right /= np.linalg.norm(right)
    down = np.cross(view, right)
    if down[1] < 0:
        down, right = -down, -right
    return np.stack([right, down, view])


def draw(Xa, Xb, C):
    """outline pixels and their motion to the next frame: nearest point per pixel, holes closed, motion smoothed"""
    pa, pb = Xa @ C.T, Xb @ C.T
    ua, fl = pa[:, :2] * PX, (pb[:, :2] - pa[:, :2]) * PX
    lo = np.floor(ua.min(0)) - 12
    ij = np.floor(ua - lo).astype(np.int64)
    W, Hh_ = ij[:, 0].max() + 13, ij[:, 1].max() + 13
    lin = ij[:, 1] * W + ij[:, 0]
    order = np.lexsort((pa[:, 2], lin))
    first = np.r_[True, lin[order][1:] != lin[order][:-1]]
    sel = order[first]
    mask = np.zeros((Hh_, W), bool)
    fg = np.zeros((Hh_, W, 2))
    mask[ij[sel, 1], ij[sel, 0]] = True
    fg[ij[sel, 1], ij[sel, 0]] = fl[sel]
    closed = binary_closing(mask, iterations=1)
    if (closed & ~mask).any():  # closed holes take the nearest drawn pixel's motion
        idx = distance_transform_edt(~mask, return_distances=False, return_indices=True)
        fg = fg[idx[0], idx[1]]
        fg[~closed] = 0
    mask = closed
    if BLUR > 0:
        s = BLUR * np.sqrt(mask.sum())
        fg = np.stack([gaussian_filter(fg[..., k], s) for k in range(2)], -1)
    r, c = np.nonzero(mask)
    return np.stack([c + lo[0], r + lo[1]], 1).astype(np.float64), fg[r, c]


# ---- the windows: (track, start, spacing, azimuth, elevation)
if os.environ.get("WINDOWS"):
    wins = json.load(open(os.environ["WINDOWS"]))
else:
    cand = []
    for k in use:
        tr = tracks[k]
        n = len(tr["bones"])
        for t0 in range(0, n - NPAIR - 1, 4):
            for gp in (1, 2, 3, 4):
                if t0 + NPAIR * gp >= n:
                    break
                d = np.linalg.norm(np.diff(tr["root"][t0:t0 + NPAIR * gp + 1:gp][:, [0, 2]], axis=0), axis=1).mean() / SIZE0
                if d >= 0.7 * TARGET or gp == 4:  # the smallest spacing whose travel per pair reaches the target
                    break
            if t0 + NPAIR * gp >= n:
                continue
            speed = d / (0.1 * gp)
            net = np.linalg.norm(tr["root"][t0 + NPAIR * gp][[0, 2]] - tr["root"][t0][[0, 2]])
            if speed >= 0.3 and net > 0.05:
                cand.append((k, t0, gp))
    pick = rng.permutation(len(cand))[:MAXWIN]
    wins = [[int(cand[i][0]), int(cand[i][1]), int(cand[i][2]), float(rng.uniform(55, 125) * rng.choice([-1, 1])),
             float(rng.uniform(5, 30))] for i in sorted(pick)]
    print("windows: %d candidates (moving at 0.3 sizes per second or more), %d drawn" % (len(cand), len(wins)), flush=True)
json.dump(wins, open(out.replace(".npz", "_windows.json"), "w"))

rows = []
for wi, (k, t0, gp, az, el) in enumerate(wins):
    tr = tracks[k]
    X = world_points(tr, t0, t0 + NPAIR * gp + 1)[::gp]
    C = camera(tr["root"][t0 + NPAIR * gp] - tr["root"][t0], az, el)
    for p in range(NPAIR):
        xy, fl = draw(X[p], X[p + 1], C)
        rows.append([wi, k, t0 + p * gp, gp, 1 if tr["kind"] == "rec" else 2] + pair_measures(xy, fl, np.zeros(2)).tolist())
    if wi % 25 == 0:
        print("  window %d of %d" % (wi, len(wins)), flush=True)
R = np.array(rows, dtype=np.float64)
np.savez(out, rows=R, cols=np.array(["win", "track", "t", "gap", "kind"] + NAMES),
         settings=json.dumps({"TARGET": TARGET, "BLUR": BLUR, "PLANT": PLANT, "SLIDE_MM": 1000 * SLIDE, "TRACKS": TRACKS, "SEED": SEED}))
tv = np.hypot(R[:, 5], R[:, 6])
print("pairs: %d in %d windows; travel per pair median %.3f sizes; size median %.0f px" % (len(R), len(wins), np.median(tv), np.median(R[:, -2])))
