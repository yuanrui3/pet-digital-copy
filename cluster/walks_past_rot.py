"""The walk-start facing fix (work/pets/PROTOCOL_CAT.md, "Smooth playback and the walk-start facing"): each generated
walk opens with 0.8 s (8 frames) of the real cat's recorded past, but the rollout export (b1_rollout.py) saved only
its positions and joints, so the page faced the cat along the walk's first predicted heading for those frames. Here
each walk's past is found in the recordings (paths.bin: the 8 recorded positions that match best) and the recorded
headings of those frames are added to walks.json as "past_rot". Then, so the cat does not turn suddenly at the
hand-over, the first K=10 predicted headings ease from the last recorded heading to the predicted ones ("lead_rot"; a
world turn shrinking smoothly to none, in front of the prediction). Prints the checks A1 and A2. Standard library only.
usage: python3 walks_past_rot.py DATA_DIR [--write]   (DATA_DIR: the page's data folder: paths.bin/.json, walks.json)"""
import json
import math
import os
import struct
import sys

d = sys.argv[1]
pm = json.load(open(os.path.join(d, "paths.json")))
buf = open(os.path.join(d, "paths.bin"), "rb").read()
blob = {b["name"]: b for b in pm["blobs"]}


def floats(name):
    b = blob[name]
    n = b["bytes"] // 4
    return struct.unpack_from("<%df" % n, buf, b["offset"])


root, head = floats("root"), floats("heading")
N = len(root) // 3
ends, s = [], 0
for v in pm["videos"]:
    s += v["frames"]
    ends.append(s)
w = json.load(open(os.path.join(d, "walks.json")))
worst_match, angles, out = 0.0, [], []
for k, run in enumerate(w["runs"]):
    past = run["past"]
    n = len(past)
    best, bj = 1e9, -1
    for j in range(N - n + 1):
        if any(j < e < j + n for e in ends):  # the past must lie within one recording
            continue
        dx = root[3 * j] - past[0][0]
        dz = root[3 * j + 2] - past[0][2]
        if dx * dx + dz * dz > 0.01:
            continue
        err = max(math.dist(root[3 * (j + i):3 * (j + i) + 3], past[i]) for i in range(n))
        if err < best:
            best, bj = err, j
    worst_match = max(worst_match, best)
    pr = [list(round(x, 4) for x in head[9 * (bj + i):9 * (bj + i) + 9]) for i in range(n)]
    r0 = run["rot"][0]
    # angle between the substituted heading (rot[0]) and the recorded one, per past frame
    for H in pr:
        tr = sum(H[3 * a + b] * r0[3 * a + b] for a in range(3) for b in range(3))
        angles.append(math.degrees(math.acos(max(-1.0, min(1.0, (tr - 1) / 2)))))
    out.append((k, bj, best, pr))
angles.sort()
print("%d walks; past found in the recordings, worst position mismatch %.4f m" % (len(out), worst_match))
print("substituted vs recorded heading over the %d past frames: median %.1f deg, 90th pct %.1f, max %.1f; over 20 deg %.0f%%" % (
    len(angles), angles[len(angles) // 2], angles[int(len(angles) * 0.9)], angles[-1], 100 * sum(a > 20 for a in angles) / len(angles)))


# ---- the hand-over: ease from the last recorded heading to the predicted ones over the first K predicted frames
K = 10


def quat(H):  # rotation matrix (row-major 9) -> unit quaternion (w, x, y, z)
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = H
    t = m00 + m11 + m22
    if t >= m00 and t >= m11 and t >= m22:
        q = (1 + t, m21 - m12, m02 - m20, m10 - m01)
    elif m00 >= m11 and m00 >= m22:
        q = (m21 - m12, 1 + m00 - m11 - m22, m01 + m10, m02 + m20)
    elif m11 >= m22:
        q = (m02 - m20, m01 + m10, 1 - m00 + m11 - m22, m12 + m21)
    else:
        q = (m10 - m01, m02 + m20, m12 + m21, 1 - m00 - m11 + m22)
    n = math.sqrt(sum(v * v for v in q))
    return tuple(v / n for v in q)


def qmat(q):
    w, x, y, z = q
    return [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y), 2 * (x * y + w * z), 1 - 2 * (x * x + z * z),
            2 * (y * z - w * x), 2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]


def mmul(A, B):
    return [sum(A[3 * i + k] * B[3 * k + j] for k in range(3)) for i in range(3) for j in range(3)]


def mT(A):
    return [A[0], A[3], A[6], A[1], A[4], A[7], A[2], A[5], A[8]]


def qpow(q, a):  # the rotation q scaled to the fraction a of its angle
    w = max(-1.0, min(1.0, q[0]))
    if w < 0:
        q, w = tuple(-v for v in q), -w
    th = math.acos(w)
    if th < 1e-9:
        return (1.0, 0.0, 0.0, 0.0)
    s = math.sin(th)
    return (math.cos(a * th),) + tuple(v / s * math.sin(a * th) for v in q[1:])


def angle(A, B):
    tr = sum(A[i] * B[i] for i in range(9))
    return math.degrees(math.acos(max(-1.0, min(1.0, (tr - 1) / 2))))


sm = lambda x: x * x * (3 - 2 * x)  # noqa: E731
lead, shown = {}, {}
for k, bj, best, pr in out:
    rot = w["runs"][k]["rot"]
    dq = quat(mmul(pr[-1], mT(rot[0])))  # world turn taking the first predicted heading onto the last recorded one
    lr = []
    for i in range(min(K, len(rot))):
        r = qmat(qpow(dq, 1 - sm((i + 1) / (K + 1))))
        lr.append([round(v, 4) for v in mmul(r, rot[i])])
    lead[k] = lr
    shown[k] = pr + lr + [list(x) for x in rot[len(lr):]]
# A1: past frames equal the recorded headings
a1 = max(angle(H, head[9 * (bj + i):9 * (bj + i) + 9]) for k, bj, best, pr in out for i, H in enumerate(pr))
# A2: turning per frame in the first 2 s after the hand-over against the walks' own turning later on
early, later, early_old = [], [], []
for k, bj, best, pr in out:
    H = shown[k]
    n0 = len(pr)
    for i in range(1, len(H)):
        a = angle(H[i - 1], H[i])
        (early if n0 <= i < n0 + 20 else later if i >= n0 + 20 else []).append(a)
    old = [w["runs"][k]["rot"][0]] * n0 + [list(x) for x in w["runs"][k]["rot"]]
    early_old += [angle(old[i - 1], old[i]) for i in range(n0, n0 + 20)]
later.sort()
p99 = later[int(len(later) * 0.99)]
print("A1: past frames vs recorded heading, worst %.3f deg (limit 0.1): %s" % (a1, "pass" if a1 <= 0.1 else "FAIL"))
print("A2: first 2 s after the hand-over, worst turn per frame %.1f deg (before the fix %.1f); the walks' own p99 later on %.1f deg: %s" % (
    max(early), max(early_old), p99, "pass" if max(early) <= p99 else "FAIL"))
print("    share of early frames over that p99: %.1f%%" % (100 * sum(a > p99 for a in early) / len(early)))

if "--write" in sys.argv:
    assert worst_match < 0.002, "a walk's past did not match a recording"
    for k, bj, best, pr in out:
        w["runs"][k]["past_rot"] = pr
        w["runs"][k]["lead_rot"] = lead[k]
    json.dump(w, open(os.path.join(d, "walks.json"), "w"), separators=(",", ":"))
    print("walks.json: past_rot added to %d walks" % len(out))
