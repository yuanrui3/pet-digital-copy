"""Floor fix (work/pets/PROTOCOL_CAT.md, "Feet and pose check"): put the cat on the room scan's floor.

The page's floor height (2nd percentile of the scan's heights) lies 7 cm below the scan's surface, and against that
surface the cat's paws mostly sit inside the floor. Here: the scan's floor level = the densest 1 cm band of heights in
its bottom 30 cm; under the cat = median height of scan points within 10 cm of that level and 8 cm sideways of
Pikachu's lowest paw that is down (else its root). Each frame the cat (both cats share the lowest point) is raised or
lowered by the gap when its lowest point is within 8 cm of that surface, fading to nothing at 12 cm, smoothed over 0.5 s.
CPU only.

usage: python feet_floor.py PAGE_DIR CHECK_DIR OUT_DIR
Writes OUT_DIR/feet_floor.npz (dy per recorded and walk frame, metres, room's y down; added to the cat's place) and
feet_floor.json.
"""
import json
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feet_check import DOWN, Page  # noqa: E402
from home2_transfer import read_blobs  # noqa: E402

NEAR, FAR = 0.08, 0.12
RADIUS = 0.08
SMOOTH = 5


def stats(g):
    return {"median_cm": 100 * float(np.median(g)), "p10_cm": 100 * float(np.percentile(g, 10)),
            "p90_cm": 100 * float(np.percentile(g, 90)), "frames": int(len(g))}


def readout(s):
    ok = abs(s["median_cm"]) <= 1.5 and abs(s["p10_cm"]) <= 3 and abs(s["p90_cm"]) <= 3
    return "on the floor" if ok else "floats or sinks"


def main():
    page_dir, check_dir, out = sys.argv[1:4]
    os.makedirs(out, exist_ok=True)
    page = Page(page_dir)
    F = np.load(os.path.join(check_dir, "feet_frames.npz"))
    track, kind, low = F["track"], F["kind"], F["lowW"]
    R = np.concatenate([tr["root"] for tr in page.tracks])
    rm = json.load(open(os.path.join(page_dir, "room.json")))
    room = read_blobs(os.path.join(page_dir, "room.bin"), rm)["pos"].astype(np.float64)
    y = room[:, 1]
    yb = y[(y > page.floor - 0.30) & (y < page.floor + 0.05)]
    hist, edges = np.histogram(yb, bins=np.arange(page.floor - 0.30, page.floor + 0.05 + 1e-9, 0.01))
    level = float(edges[np.argmax(hist)] + 0.005)
    band = room[np.abs(y - level) <= 0.10]
    tree = cKDTree(band[:, [0, 2]])
    print("page floor %.3f m; scan floor level %.3f m (%.1f cm above it); %d scan points in the band" % (
        page.floor, level, 100 * (page.floor - level), len(band)), flush=True)

    # where under the cat: Pikachu's lowest paw that is down, else the root
    pawP, hP = F["pawP"], F["hP"]
    down = hP <= DOWN
    lowest = np.argmax(np.where(down, pawP[..., 1], -np.inf), 1)
    xz = np.where(down.any(1)[:, None], pawP[np.arange(len(pawP)), lowest][:, [0, 2]], R[:, [0, 2]])
    local = np.full(len(xz), np.nan)
    for i, nb in enumerate(tree.query_ball_point(xz, RADIUS)):
        if len(nb) >= 5:
            local[i] = np.median(band[nb, 1])
    gap = local - low  # positive: the cat's lowest point above the scan's surface (y down)
    ramp = np.clip((FAR - np.abs(gap)) / (FAR - NEAR), 0, 1)
    raw = np.where(np.isfinite(gap), gap * ramp, 0.0)
    dy = np.zeros_like(raw)
    starts = np.r_[0, np.where(np.diff(track) != 0)[0] + 1, len(track)]
    k = np.ones(SMOOTH) / SMOOTH
    for a, b in zip(starts[:-1], starts[1:]):
        x = np.pad(raw[a:b], SMOOTH // 2, mode="edge")
        dy[a:b] = np.convolve(x, k, mode="valid")
    res = {"page_floor_m": page.floor, "scan_floor_level_m": level, "scan_points_in_band": int(len(band)),
           "frames_with_scan_under": float(np.isfinite(local).mean())}
    for kk in ("rec", "walk"):
        m = (kind == kk) & np.isfinite(gap) & (np.abs(gap) <= 0.10)
        before, after = stats(gap[m]), stats(gap[m] - dy[m])
        res[kk] = {"share_near_floor": float(m[kind == kk].mean()), "before": before, "after": after,
                   "readout_before": readout(before), "readout_after": readout(after),
                   "correction_cm_median_abs": 100 * float(np.median(np.abs(dy[kind == kk])))}
    nrec = int((kind == "rec").sum())
    np.savez_compressed(os.path.join(out, "feet_floor.npz"), dy=dy, rec_dy=dy[:nrec], walk_dy=dy[nrec:], gap=gap)
    json.dump(res, open(os.path.join(out, "feet_floor.json"), "w"), indent=1)
    print(json.dumps(res, indent=1), flush=True)


if __name__ == "__main__":
    main()
