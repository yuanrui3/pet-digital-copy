"""The video-model judge (work/pets/PROTOCOL_CAT.md, "A video-model judge"): our cat's silhouette clips.

The page's cat (version 22's shape and grafted head, as judge_ours.py) is posed at 30 steps a second as the page shows
it, and drawn as a white silhouette on black from a still, level, orthographic camera beside its path, on the ruler's
side-view windows (experiments_judge/ours/walk_shown_windows.json: track, first 0.1 s frame, -, azimuth, elevation).
  - clip: 16 frames, one every 4 steps (7.5 a second, 2.0 s), from the window's first frame (moved earlier if the walk
    ends first);
  - crop: one still square per clip round everything the cat covers in it (1.15 times the larger side), drawn at
    512 px and reduced to 256 (area averaging), so the cat's travel stays in the picture as it does in the real clips
    (made still by removing the camera's pan);
  - SOURCE: a gait_dense.npz (gait_cycle.py DENSE=1, the walking cycle) or "v22" (version 22's walks as the page
    showed them: planted-paw bones at the 0.1 s samples, everything held for 3 steps);
  - DEFECT (the J2 ladder, on the source): hold:STEPS (every pose, place and facing held this many steps), jitter:DEG
    (each bone turned about its joint by a random axis and a normal angle of this many degrees sd, new each frame,
    the same draws scaled for every size), back (the clip played backwards), stiff:SHARE (the leg bones blended
    elementwise this share of the way to version 22's, those interpolated between the 0.1 s samples).
CPU only.
usage: python jepa_ours.py PAGE_DIR IK_DIR FLOOR_DIR GRAFT_NPZ WINDOWS_JSON OUT_NPZ
Writes OUT_NPZ: clips (windows x 16 x 256 x 256, uint8), win (window numbers used), and a json of settings."""
import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feet_check import Page  # noqa: E402
from feet_ik import CHAINS, bend, joints_of  # noqa: E402
from home2_transfer import lbs, read_blobs  # noqa: E402

page_dir, ik_dir, floor_dir, graft, win_path, out = sys.argv[1:7]
SOURCE = os.environ.get("SOURCE", "v22")
DEFECT = os.environ.get("DEFECT", "none")
NF, EVERY, RES, MARGIN = 16, 4, 256, 1.15
SPAN = EVERY * (NF - 1)                     # 60 steps

pg = Page(page_dir)
g = np.load(graft)
Ps, Hh, Fh = g["Ps"].astype(np.float64), g["H"].astype(np.float64), g["Fh"].astype(np.int64)
hidx, hw = g["hidx"].astype(np.int64), g["hw"].astype(np.float64)
FACES = np.concatenate([pg.fp, Fh + len(Ps)])
im = json.load(open(os.path.join(ik_dir, "feet_ik.json")))
IK = read_blobs(os.path.join(ik_dir, "feet_ik.bin"), im)
Jr = np.array(im["rest_joints"], dtype=np.float64)
FL = read_blobs(os.path.join(floor_dir, "feet_floor.bin"), json.load(open(os.path.join(floor_dir, "feet_floor.json"))))

# the judge_ours track numbering (recordings and walks in page order), each walk with its place in the walks' data
tracks, nwalk = [], 0
for tr in pg.tracks:
    t = dict(tr)
    if tr["kind"] != "rec":
        t["base"] = nwalk
        nwalk += len(tr["bones"])
    tracks.append(t)
DN = np.load(SOURCE) if SOURCE != "v22" else None
dn_at = {}
if DN is not None:
    off = 0
    for b, n in zip(DN["base"], DN["n"]):
        dn_at[int(b)] = (off, int(n))
        off += 3 * int(n)
LEGB = CHAINS.reshape(-1)


def v22_track(tr, held=True):
    """version 22's walk at 30 steps: planted-paw bones at the samples, held (as shown) or blended between them"""
    n, a = len(tr["bones"]), tr["base"]
    T = bend(tr["bones"].astype(np.float64), Jr, IK["walk"][a:a + n].astype(np.float64))
    dy = FL["walk_dy"][a:a + n].astype(np.float64)
    if held:
        rep = np.repeat(np.arange(n), 3)
        return T[rep], tr["root"][rep].astype(np.float64), tr["H"][rep].astype(np.float64), dy[rep]
    t = np.repeat(np.arange(n), 3)
    f = np.tile(np.arange(3) / 3, n)
    t1 = np.minimum(t + 1, n - 1)
    return T[t] * (1 - f)[:, None, None, None] + T[t1] * f[:, None, None, None], None, None, None


def track_steps(tr):
    if DN is None:
        return v22_track(tr)
    o, n = dn_at[tr["base"]]
    sl = slice(o, o + 3 * n)
    return (DN["bones"][sl].astype(np.float64), DN["root"][sl].astype(np.float64), DN["H"][sl].astype(np.float64),
            DN["dy"][sl].astype(np.float64))


def camera(travel, az, el):
    """rows: image right, image down, view direction (judge_ours.py's camera)"""
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


def axis_angle(ax, ang):
    """(...,3) unit axes, (...) angles (radians) -> (...,3,3)"""
    K = np.zeros(ax.shape[:-1] + (3, 3))
    K[..., 0, 1], K[..., 0, 2], K[..., 1, 2] = -ax[..., 2], ax[..., 1], -ax[..., 0]
    K = K - np.swapaxes(K, -1, -2)
    s, c = np.sin(ang)[..., None, None], np.cos(ang)[..., None, None]
    return np.eye(3) + s * K + (1 - c) * (K @ K)


def silhouettes(X2):
    """(16, V, 2) image-plane points (m) -> (16, 256, 256) uint8, one still square round all of them"""
    lo, hi = X2.reshape(-1, 2).min(0), X2.reshape(-1, 2).max(0)
    side = MARGIN * (hi - lo).max()
    c = (lo + hi) / 2
    big = 2 * RES
    out_ = np.zeros((len(X2), RES, RES), np.uint8)
    for k, P in enumerate(X2):
        q = ((P - (c - side / 2)) / side * big * 16).astype(np.int32)  # 4 bits below the pixel
        img = np.zeros((big, big), np.uint8)
        for tri in q[FACES]:  # one at a time: a single fillPoly call clears where triangles overlap (even-odd rule)
            cv2.fillConvexPoly(img, tri, 255, lineType=cv2.LINE_8, shift=4)
        out_[k] = cv2.resize(img, (RES, RES), interpolation=cv2.INTER_AREA)
    return out_


wins = json.load(open(win_path))
kind, _, arg = DEFECT.partition(":")
clips, used, t_start = [], [], time.time()
cache = {}
for wi, (k, t0, _gp, az, el) in enumerate(wins):
    tr = tracks[k]
    n = len(tr["bones"])
    if 3 * n < SPAN + 1:
        continue
    if k not in cache:
        cache.clear()
        cache[k] = track_steps(tr)
        if kind == "stiff":
            cache[(k, "v22")] = v22_track(tr, held=False)[0]
    T, root, H, dy = cache[k]
    s0 = min(3 * t0, 3 * n - SPAN - 1)
    steps = s0 + EVERY * np.arange(NF)
    if kind == "hold":
        h = int(arg)
        steps = (steps // h) * h
    if kind == "back":
        steps = steps[::-1]
    Tc = T[steps].copy()
    if kind == "stiff":
        a = float(arg)
        Tc[:, LEGB] = (1 - a) * Tc[:, LEGB] + a * cache[(k, "v22")][steps][:, LEGB]
    if kind == "jitter":
        rng = np.random.default_rng(1000 + wi)  # the same draws for every size
        ax = rng.normal(size=(NF, Tc.shape[1], 3))
        ax /= np.linalg.norm(ax, axis=-1, keepdims=True)
        R = axis_angle(ax, np.radians(float(arg)) * rng.normal(size=(NF, Tc.shape[1])))
        J = joints_of(Tc, Jr)
        Tc[..., :3] = R @ Tc[..., :3]
        Tc[..., 3] = np.einsum("fbij,fbj->fbi", R, Tc[..., 3] - J) + J
    C = camera(tr["root"][min(n - 1, (s0 + SPAN) // 3)] - tr["root"][s0 // 3], az, el)
    X2 = []
    for f, st in enumerate(steps):
        X = np.concatenate([lbs(Ps, pg.ip, pg.wp, Tc[f]), lbs(Hh, hidx, hw, Tc[f])])
        Y = X @ H[st].T + root[st]
        Y[:, 1] += dy[st]
        X2.append((Y @ C.T)[:, :2])
    clips.append(silhouettes(np.array(X2)))
    used.append(wi)
    if wi % 50 == 0:
        print("  window %d of %d (%.0f s)" % (wi, len(wins), time.time() - t_start), flush=True)
np.savez_compressed(out, clips=np.array(clips), win=np.array(used))
json.dump({"SOURCE": SOURCE, "DEFECT": DEFECT, "windows": len(used), "frames": NF, "every_steps": EVERY, "res": RES,
           "margin": MARGIN, "seconds": round(time.time() - t_start, 1)}, open(out.replace(".npz", ".json"), "w"))
print("clips: %d of %d windows, %.0f s" % (len(used), len(wins), time.time() - t_start), flush=True)
