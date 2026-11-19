"""The judge without ratings (work/pets/PROTOCOL_CAT.md, "Building the judge"): the measures shared by the real clips
(judge_real.py) and our cat (judge_ours.py). A pair is two consecutive frames; everything comes from the cat's outline
pixels and their image motion, in units of the cat's size (the square root of its outline area)."""
import numpy as np

STRIP = 0.08   # the lowest strip of the outline (share of its height): the paws on the ground
LEG = 0.40     # the lower part of the outline: the legs
TORSO = 0.50   # the upper part: the body
NAMES = ["tx", "ty", "slip10", "slip25", "leg", "torso", "a_fore", "a_hind", "w", "h", "size_px", "n"]


def pair_measures(xy, flow, ground, pixel_area=1.0):
    """xy (n,2): the outline's pixels (x right, y down); flow (n,2): their image motion to the next frame; ground (2,):
    the ground's image motion; all in pixels. Returns the NAMES values: lengths in sizes.
      tx, ty   the body's travel over the ground (mean motion minus the ground's)
      slip10   in the lowest strip, the 10th percentile of the speed over the ground (a planted paw: about 0)
      slip25   the same, 25th percentile
      leg      mean motion of the leg part relative to the body's mean motion
      torso    the same for the upper part
      a_fore, a_hind  mean motion of the leg part ahead of / behind the middle, relative to the body, along the travel
      w, h     the outline's extent (2nd to 98th percentile)"""
    n = len(xy)
    size = np.sqrt(n * pixel_area)
    x, y = xy[:, 0], xy[:, 1]
    xlo, xhi = np.percentile(x, [2, 98])
    ylo, yhi = np.percentile(y, [2, 98])
    w, h = xhi - xlo, yhi - ylo
    body = flow.mean(0)
    v = body - ground
    internal = flow - body
    strip = y > yhi - STRIP * h
    leg = y > ylo + (1 - LEG) * h
    top = y < ylo + TORSO * h
    sp = np.linalg.norm(flow[strip] - ground, axis=1) if strip.sum() >= 5 else np.array([np.nan])
    vn = np.linalg.norm(v)
    vh = v / vn if vn > 1e-9 else np.array([1.0, 0.0])
    along = (x - x.mean()) * vh[0] + (y - y.mean()) * vh[1]
    fore, hind = leg & (along > 0), leg & (along <= 0)
    a = internal @ vh
    out = [v[0], v[1], np.nanpercentile(sp, 10), np.nanpercentile(sp, 25),
           np.linalg.norm(internal[leg], axis=1).mean() if leg.any() else np.nan,
           np.linalg.norm(internal[top], axis=1).mean() if top.any() else np.nan,
           a[fore].mean() if fore.sum() >= 5 else np.nan, a[hind].mean() if hind.sum() >= 5 else np.nan, w, h]
    return np.array([o / size for o in out] + [size, n], dtype=np.float64)
