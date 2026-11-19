"""Phase 0, P0-1 (work/pets/PROTOCOL_CAT.md, "Restart, Phase 0"): the generated walks' sudden turns where the behaviour
model's 5.6 s windows meet. For each seam step (and the hand-over from the recorded past), and for other steps of the
walks and of the recordings as baselines: the turn in that 0.1 s step, the cat's speed over the 0.5 s before, and the
angle between its facing before the step and the direction the path takes over the next 1 s. Standard library only.
usage: python3 p0_seams.py DATA_DIR   (the page's data folder: walks.json, paths.bin/.json)"""
import json
import math
import os
import struct
import sys

d = sys.argv[1]
w = json.load(open(os.path.join(d, "walks.json")))
pm = json.load(open(os.path.join(d, "paths.json")))
buf = open(os.path.join(d, "paths.bin"), "rb").read()
blob = {b["name"]: b for b in pm["blobs"]}


def floats(name):
    b = blob[name]
    return struct.unpack_from("<%df" % (b["bytes"] // 4), buf, b["offset"])


def angle(A, B):
    tr = sum(A[i] * B[i] for i in range(9))
    return math.degrees(math.acos(max(-1.0, min(1.0, (tr - 1) / 2))))


def facing(H):  # the cat faces -z in its own frame; the room's y points down, so the floor is x, z
    return (-H[2], -H[8])


def ang2(u, v):
    nu, nv = math.hypot(*u), math.hypot(*v)
    if nu < 1e-9 or nv < 1e-9:
        return None
    c = (u[0] * v[0] + u[1] * v[1]) / (nu * nv)
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def step_rows(H, P, kinds):
    """H: headings (9 each), P: positions (3 each) per frame; kinds(i) -> label for the step i-1 -> i"""
    rows = []
    for i in range(5, len(H) - 10):
        turn = angle(H[i - 1], H[i])
        speed = math.hypot(P[i - 1][0] - P[i - 6][0], P[i - 1][2] - P[i - 6][2]) / 0.5
        ahead = (P[i + 10][0] - P[i][0], P[i + 10][2] - P[i][2])
        moves = math.hypot(*ahead) > 0.05
        rows.append((kinds(i), turn, speed, ang2(facing(H[i - 1]), ahead) if moves else None,
                     ang2(facing(H[i]), ahead) if moves else None))
    return rows


rows = []
for run in w["runs"]:
    past, pos, rot = run["past"], run["pos"], run["rot"]
    n0 = len(past)
    if "past_rot" not in run:  # the recorded past's headings: found in the recordings (walks_past_rot.py)
        pass
    H = [run.get("past_rot", [rot[0]] * n0)[k] for k in range(n0)] + [list(x) for x in rot]
    P = [list(x) for x in past] + [list(x) for x in pos]
    rows += step_rows(H, P, lambda i, n0=n0: "hand-over" if i == n0 else "seam" if i > n0 and (i - n0) % 56 == 0 else "walk")
root, head = floats("root"), floats("heading")
s = 0
for v in pm["videos"]:
    n = v["frames"]
    H = [head[9 * (s + k):9 * (s + k) + 9] for k in range(n)]
    P = [root[3 * (s + k):3 * (s + k) + 3] for k in range(n)]
    rows += step_rows(H, P, lambda i: "recording")
    s += n


def pct(x, q):
    x = sorted(x)
    return x[min(len(x) - 1, int(q * (len(x) - 1) + 0.5))] if x else float("nan")


print("hand-over rows use the recorded past headings: %s" % ("yes" if all("past_rot" in r for r in w["runs"]) else
      "no (walks.json has no past_rot; the hand-over turn here is from the page's substituted heading, so it is 0)"))
print("%-10s %6s  %5s %5s %5s %5s  %6s" % ("steps", "n", "p50", "p95", "p99", "max", ">45deg"))
for k in ("seam", "hand-over", "walk", "recording"):
    t = [r[1] for r in rows if r[0] == k]
    if t:
        print("%-10s %6d  %5.1f %5.1f %5.1f %5.1f  %5.1f%%" % (k, len(t), pct(t, .5), pct(t, .95), pct(t, .99), max(t),
                                                          100 * sum(a > 45 for a in t) / len(t)))
seam = [r for r in rows if r[0] == "seam"]
print("\nseams: turn against the speed before the seam")
for lo, hi in ((0, 0.05), (0.05, 0.15), (0.15, 0.3), (0.3, 9)):
    t = [r[1] for r in seam if lo <= r[2] < hi]
    if t:
        print("  speed %.2f-%.2f m/s: %4d seams, turn p50 %5.1f, share >45deg %5.1f%%" % (lo, hi, len(t), pct(t, .5), 100 * sum(a > 45 for a in t) / len(t)))
print("seams: turn against the angle between the facing before the seam and the new path (path moving >5 cm in 1 s)")
for lo, hi in ((0, 45), (45, 90), (90, 135), (135, 181)):
    t = [r[1] for r in seam if r[3] is not None and lo <= r[3] < hi]
    if t:
        print("  path at %3d-%3d deg to the facing: %4d seams, turn p50 %5.1f, share >45deg %5.1f%%" % (lo, hi, len(t), pct(t, .5), 100 * sum(a > 45 for a in t) / len(t)))
print("  path not moving (<5 cm in 1 s): %d seams, turn p50 %.1f" % (sum(r[3] is None for r in seam), pct([r[1] for r in seam if r[3] is None], .5)))
for k in ("seam", "walk", "recording"):
    a0 = [r[3] for r in rows if r[0] == k and r[3] is not None]
    a1 = [r[4] for r in rows if r[0] == k and r[4] is not None]
    print("%-9s facing vs path ahead: before the step p50 %5.1f deg (p90 %5.1f); after it p50 %5.1f (p90 %5.1f)" % (
        k, pct(a0, .5), pct(a0, .9), pct(a1, .5), pct(a1, .9)))
