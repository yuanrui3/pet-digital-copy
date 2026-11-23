"""The walking cycle on the copy's walks (work/pets/PROTOCOL_CAT.md, "A built-in walking cycle on the copy's walks").

Per copy run (all of them), 30 steps a second (SUB = 3 per 0.1 s sample). The body (place, facing, spine, head, tail)
is the behaviour model's; only the legs change:
  - the body is lowered (smoothed over 0.5 s) so each leg is at EXT of its full length at mid-stance;
  - a paw that is down stays exactly where it landed, on the ground;
  - it lifts when its shoulder or hip has carried on TRIG past it (or it can no longer reach), at most two paws in the
    air and never both of a pair, the paw furthest behind first; it swings for SWING_STEPS (0.3 s), 2.5 to 3.5 cm up,
    to land under its shoulder or hip as it will be then, plus the travel of the next 0.3 s (at most LEAD_CAP);
  - each leg is bent at its own joints to put its wrist or hock where the paw needs it: the top joint stays, the
    middle joint bends to the side of POLE (elbows towards the tail, knees towards the head), the paw bone moves
    without turning, the segments may lengthen by up to 5% (as version 18-22's planted paws).
The page interpolates between the 0.1 s samples as this script does: place and floor shift linearly, facing by the
shortest turn (held at the earlier sample when the turn is over CUT_DEG), bones elementwise; the paw targets are in
the room, so they do not depend on that interpolation.
CPU only.

usage: python gait_cycle.py PAGE_DIR IK_DIR FLOOR_DIR OUT_DIR
Writes OUT_DIR/gait.bin + gait.json (for the page: per walk frame, SUB steps and 4 legs, the wrist or hock target
minus the cat's place at that frame, room axes; per step the floor shift), gait_bones.npz (the bent bones and floor
shift at the 0.1 s samples, for the checks and the ruler), gait_stats.json and gait_test.json (bent bones at a few
steps, to check the page's solver)."""
import json
import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feet_check import Page, skin  # noqa: E402
from feet_ik import CHAINS, half_turn, joints_of, rot_between  # noqa: E402
from home2_transfer import pack, read_blobs  # noqa: E402

page_dir, ik_dir, floor_dir, out = sys.argv[1:5]
os.makedirs(out, exist_ok=True)
SUB = 3
EXT = np.array([float(x) for x in os.environ.get("EXT", "0.95,0.95,0.95,0.95").split(",")])
LOWER_BY = os.environ.get("LOWER_BY", "mean")  # body lowering: the mean of what the legs need, or the least ("min")
TRIG = float(os.environ.get("TRIG", 0.05))
SLANT = float(os.environ.get("SLANT", 1.0))  # share of the model's own fore-aft paw offset kept at mid-stance
MAX_BEND = float(os.environ.get("MAX_BEND", 70.0))  # the middle joint bends at most this far from straight (degrees)
MIN_STEP = 0.02           # no step shorter than this (it would only twitch)
SWING_STEPS = 9           # 0.3 s
LEAD_STEPS, LEAD_CAP = 9, 0.04
LIFT_BASE, LIFT_PER_M = 0.015, 0.4
LIFT_MIN, LIFT_MAX = float(os.environ.get("LIFT_MIN", 0.025)), float(os.environ.get("LIFT_MAX", 0.035))
# from real cats' video (gait_stats.py; PROTOCOL_CAT.md "Real cats' gait from video"): swing time by speed, leg order
SWING_SPEED = os.environ.get("SWING_SPEED", "0") == "1"  # 0.30 s up to 0.35 body lengths a second, 0.20 s from 0.65
SEQ = os.environ.get("SEQ", "0") == "1"                  # lateral sequence: a front paw steps after its side's hind
BODY_LEN = 0.33                                          # our cat, neck base to tail base (m)
BODY = os.environ.get("BODY")                            # body_stats.json: head and shoulders move with the steps
HEAD_BONE = 9                                            # the bone that mainly moves the head and neck in the skinning
HEAD_JOINT = 3                                           # the head joint (joint 4), the nod's pivot as bone 9 carries it
# bones the leg bending leaves alone (bone 9 moves the head); PROTECT="" rebuilds versions 23 and 24 (g5, g9)
PROTECT = tuple(int(x) for x in os.environ.get("PROTECT", "9").split(",") if x.strip())
# for the video-model judge (PROTOCOL_CAT.md, "A video-model judge"): DENSE=1 also writes gait_dense.npz, the bent
# bones, place, facing and floor shift at every 30 Hz step of the free walks, as the page shows them; SLIDE_MM (the
# defect ladder) moves each paw that is down forward by this many mm per 0.1 s since it landed (at most 1 s), the
# step timing unchanged
DENSE = os.environ.get("DENSE") == "1"
SLIDE = float(os.environ.get("SLIDE_MM", 0)) / 1000
LEVEL = int(os.environ.get("LEVEL", "0"))               # 1: pitch the body so front and hind legs bend alike; 2: pitch and roll
MAX_PITCH = np.radians(float(os.environ.get("MAX_PITCH", 10)))
STRETCH = 1.05
MIN_STANCE = 3            # steps a paw stays down before it may lift again
CUT_DEG = 45.0
FLOOR_NEAR = 0.08
POLE = np.array([[0, 0, 1.0], [0, 0, 1.0], [0, 0, -1.0], [0, 0, -1.0]])  # cat's frame: z towards the tail
PARTNER = (1, 0, 3, 2)
TOPS, MIDS, WRISTS = CHAINS[:, 0], CHAINS[:, 1], CHAINS[:, 2]


def quat_wxyz(R):
    q = Rotation.from_matrix(R).as_quat()  # x, y, z, w
    return np.c_[q[:, 3], q[:, :3]]


def mat_of(q):
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack([np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
                     np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
                     np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1)], -2)


def interp(n, r, H, dy, T):
    """the page's 30-step interpolation of place, facing, floor shift and bones -> per step (n * SUB)"""
    t = np.repeat(np.arange(n), SUB)
    f = np.tile(np.arange(SUB) / SUB, n)
    t1 = np.minimum(t + 1, n - 1)
    rk = r[t] * (1 - f)[:, None] + r[t1] * f[:, None]
    dyk = dy[t] * (1 - f) + dy[t1] * f
    Tk = T[t] * (1 - f)[:, None, None, None] + T[t1] * f[:, None, None, None]
    q = quat_wxyz(H)
    q0, q1 = q[t], q[t1].copy()
    dot = (q0 * q1).sum(1)
    q1[dot < 0] *= -1
    dot = np.abs(dot)
    ang = 2 * np.degrees(np.arccos(np.clip(dot, -1, 1)))
    th = np.arccos(np.clip(dot, -1, 1))
    small = th < 1e-6
    s0 = np.where(small, 1 - f, np.sin((1 - f) * th) / np.where(small, 1, np.sin(th)))
    s1 = np.where(small, f, np.sin(f * th) / np.where(small, 1, np.sin(th)))
    qk = s0[:, None] * q0 + s1[:, None] * q1
    qk /= np.linalg.norm(qk, axis=1, keepdims=True)
    Hk = mat_of(qk)
    hold = ang > CUT_DEG
    Hk[hold] = H[t[hold]]
    return t, rk, Hk, dyk, Tk


def two_bone_pole(Hp, Kp, Wp, tgt, pole):
    """(n,3) top, middle, wrist joints, the wrist target and the bend side -> new middle and wrist, R1, R2, lengthening"""
    a = np.linalg.norm(Kp - Hp, axis=-1)
    b = np.linalg.norm(Wp - Kp, axis=-1)
    dv = tgt - Hp
    dn = np.maximum(np.linalg.norm(dv, axis=-1), 1e-9)
    u = dv / dn[..., None]
    s = np.clip(dn / (0.999 * (a + b)), 1.0, STRETCH)
    a2, b2 = a * s, b * s
    dmin = np.sqrt(a2 * a2 + b2 * b2 + 2 * a2 * b2 * np.cos(np.radians(MAX_BEND)))  # the reach at the bend limit
    d = np.clip(dn, np.maximum(np.abs(a2 - b2) + 1e-3, dmin), 0.999 * (a2 + b2))
    pl = pole - (pole * u).sum(-1, keepdims=True) * u
    own = (Kp - Hp) - ((Kp - Hp) * u).sum(-1, keepdims=True) * u
    pl = np.where((np.linalg.norm(pl, axis=-1) < 1e-4)[..., None], own, pl)
    v = pl / np.maximum(np.linalg.norm(pl, axis=-1), 1e-9)[..., None]
    cosA = np.clip((a2 * a2 + d * d - b2 * b2) / (2 * a2 * d), -1, 1)
    sinA = np.sqrt(1 - cosA ** 2)
    K2 = Hp + a2[..., None] * (cosA[..., None] * u + sinA[..., None] * v)
    W2 = Hp + d[..., None] * u
    R1 = rot_between((Kp - Hp) / a[..., None], (K2 - Hp) / a2[..., None])
    R2 = rot_between((Wp - Kp) / b[..., None], (W2 - K2) / b2[..., None])
    return K2, W2, R1, R2, s


def bend_to(T, Jr, tgt):
    """T (n,25,3,4), wrist targets tgt (n,4,3) in the cat's frame -> bent bones, lengthening (n,4)"""
    T2 = T.copy()
    J = joints_of(T, Jr)
    sv = np.ones((len(T), 4))
    for p, (bt, bm, bw, bp) in enumerate(CHAINS):
        Hp, Kp, Wp, Pp = J[:, bt], J[:, bm], J[:, bw], J[:, bp]
        K2, W2, R1, R2, s = two_bone_pole(Hp, Kp, Wp, tgt[:, p], np.broadcast_to(POLE[p], Hp.shape))
        sv[:, p] = s
        Rm = R1 @ half_turn(np.swapaxes(R1, -1, -2) @ R2)
        for b, Rd, c, c2 in ((bt, R1, Hp, Hp), (bm, Rm, Kp, K2), (bw, half_turn(R2), Wp, W2),
                             (bp, np.broadcast_to(np.eye(3), R1.shape), Pp, Pp + (W2 - Wp))):
            if b in PROTECT:
                continue
            T2[:, b, :, :3] = Rd @ T[:, b, :, :3]
            T2[:, b, :, 3] = np.einsum("nij,nj->ni", Rd, T[:, b, :, 3] - c) + c2
    return T2, sv


def to_room(Xc, H, r, dy):
    Y = np.einsum("nij,nmj->nmi", H, Xc) + r[:, None]
    Y[..., 1] += dy[:, None]
    return Y


def axes(H):
    """(n,3,3) facing -> the cat's lateral (+x) and forward (-z) directions, levelled in the room"""
    lat, fwd = H[:, :, 0].copy(), -H[:, :, 2].copy()
    lat[:, 1] = fwd[:, 1] = 0
    return lat / np.linalg.norm(lat, axis=1, keepdims=True), fwd / np.linalg.norm(fwd, axis=1, keepdims=True)


def nominal(G, H, ground):
    """where each paw stands at mid-stance: under its top joint, the model's own sideways offset, SLANT of its fore-aft
    offset; on the ground"""
    lat, fwd = axes(H)
    N = G + O_NOM[None, :, 0, None] * lat[:, None] + SLANT * O_NOM[None, :, 1, None] * fwd[:, None]
    N[..., 1] = ground[:, None] - DEPTH[None]
    return N


def rx(th):
    """(n,) angles -> (n,3,3) turns about the cat's x axis (y down, z towards the tail; + lowers the front)"""
    R = np.zeros((len(th), 3, 3))
    R[:, 0, 0] = 1
    R[:, 1, 1], R[:, 1, 2], R[:, 2, 1], R[:, 2, 2] = np.cos(th), -np.sin(th), np.sin(th), np.cos(th)
    return R


def rz(ph):
    """(n,) angles -> (n,3,3) turns about the cat's z axis (+ lowers the +x side)"""
    R = np.zeros((len(ph), 3, 3))
    R[:, 2, 2] = 1
    R[:, 0, 0], R[:, 0, 1], R[:, 1, 0], R[:, 1, 1] = np.cos(ph), -np.sin(ph), np.sin(ph), np.cos(ph)
    return R


def level(T, th, c, ph=None):
    """every bone turned rigidly by th about the cat's x axis, then ph about its z axis, through c (n,3), as the
    page does"""
    R = rx(th) if ph is None else rz(ph) @ rx(th)
    T2 = np.empty_like(T)
    T2[..., :3] = np.einsum("nij,nbjk->nbik", R, T[..., :3])
    T2[..., 3] = np.einsum("nij,nbj->nbi", R, T[..., 3] - c[:, None]) + c[:, None]
    return T2


def level_pts(X, th, c, ph=None):
    R = rx(th) if ph is None else rz(ph) @ rx(th)
    return np.einsum("nij,nmj->nmi", R, X - c[:, None]) + c[:, None]


def head_turn(T, ang):
    """the head bone turned by ang (n,) about the cat's x axis through its own joint (+ nose down), as the page does"""
    T2 = T.copy()
    R, b = rx(ang), HEAD_BONE
    p = np.einsum("nij,j->ni", T[:, b, :, :3], Jr[HEAD_JOINT]) + T[:, b, :, 3]
    T2[:, b, :, :3] = R @ T[:, b, :, :3]
    T2[:, b, :, 3] = np.einsum("nij,nj->ni", R, T[:, b, :, 3] - p) + p
    return T2


def stride_phase(lifts, K):
    """phase over each stride of the reference hind paw (lift to next lift), and how much body motion to give: full for
    strides up to 1.6 s, none at 2.5 s and above (the cat is standing), smoothed over 0.3 s"""
    phi, amp = np.zeros(K), np.zeros(K)
    for a, b in zip(lifts[:-1], lifts[1:]):
        per = (b - a) / 30.0
        if per > 2.5:
            continue
        idx = np.arange(a, b)
        phi[idx] = (idx - a) / (b - a)
        amp[idx] = np.clip((2.5 - per) / 0.9, 0, 1)
    return phi, np.convolve(np.r_[np.zeros(4), amp, np.zeros(4)], np.ones(9) / 9, mode="valid")


def wave(w, phi):
    w = np.asarray(w, float)
    return np.interp(phi, np.r_[np.arange(len(w)) / len(w), 1.0], np.r_[w, w[0]])


def smooth(x, k=5):
    pad = np.r_[np.repeat(x[:1], k // 2), x, np.repeat(x[-1:], k // 2)]
    return np.convolve(pad, np.ones(k) / k, mode="valid")


pg = Page(page_dir)
Jr = np.array(json.load(open(os.path.join(ik_dir, "feet_ik.json")))["rest_joints"], dtype=np.float64)
BODY_W = None
if BODY:
    # BODY_SIZE=stride: scale each averaged waveform up to a single stride's size (averaging strides whose timing
    # varies shrinks it about by half)
    _bm = json.load(open(BODY))["motions"]
    _sc = lambda v: v["per_stride_half_ptp_median"] / v["half_peak_to_peak"] if os.environ.get("BODY_SIZE") == "stride" else 1.0  # noqa: E731
    BODY_W = {k: (np.array(v["waveform"]) * _sc(v)).tolist() for k, v in _bm.items() if v.get("used")}
    hv = pg.P[np.take_along_axis(pg.ip, np.argmax(pg.wp, 1)[:, None], 1)[:, 0] == HEAD_BONE]
    HEAD_R = float(np.linalg.norm(hv[np.argmin(hv[:, 2])] - Jr[HEAD_JOINT]))  # head joint to the nose
    print("body motion from %s: %s; head joint to nose %.3f m" % (BODY, sorted(BODY_W), HEAD_R), flush=True)
fm = json.load(open(os.path.join(floor_dir, "feet_floor.json")))
walk_dy = read_blobs(os.path.join(floor_dir, "feet_floor.bin"), fm)["walk_dy"].astype(np.float64)
walks = [tr for tr in pg.tracks if tr["kind"] == "walk"]
base = 0
for tr in walks:
    tr["base"] = base
    base += len(tr["bones"])
assert base == len(walk_dy), (base, len(walk_dy))

# ---- pass 1: per run, joints, paw points (the paw's landmark: mean of its vertices) and the lowest point
pre, rel, depth = [], [], []
for tr in walks:
    T, H = tr["bones"].astype(np.float64), tr["H"]
    J = joints_of(T, Jr)
    XP, _, lowP = pg.pose(T, H)
    L, _ = pg.split(XP, pg.e3p, pg.lp)
    L = L[:, :4]
    down = H[:, 1, :]
    hl = np.einsum("npi,ni->np", L, down)
    lo = np.stack([np.einsum("nvi,ni->nv", skin(T, pg.P[s], pg.ip[s], pg.wp[s]), down).max(1) for s in pg.lp[:4]], 1)
    pre.append((T, J, L, lowP))
    d = np.einsum("nij,npj->npi", H, L - J[:, TOPS])
    lat, fwd = axes(H)
    rel.append(np.stack([np.einsum("npi,ni->np", d, lat), np.einsum("npi,ni->np", d, fwd)], -1))
    depth.append(lo - hl)                              # how far the paw's lowest point is below its landmark
O_NOM = np.median(np.concatenate(rel), axis=0)         # (4,2) where the model's paw sits from its top joint: lateral, forward
DEPTH = np.median(np.concatenate(depth), axis=0)       # (4,)
print("model's paw from its top joint (room level: lateral, forward, m): %s; paw landmark above its sole: %s" % (
    np.round(O_NOM, 3).tolist(), np.round(DEPTH, 3).tolist()), flush=True)

# ---- pass 2: lower the body, step the paws, bend the legs
WR, DYK, BONES, DY10, PK, CK, RL, HK, DN = [], [], [], [], [], [], [], [], []
st = {"runs": len(walks), "steps": [0, 0, 0, 0], "forced": 0, "stance_s": [], "swing_m": [], "lowering_m": [],
      "reach_limited_steps": 0, "stance_steps": 0, "on_furniture_frames": 0}
tests, rng = [], np.random.default_rng(3)
for i, (tr, (T, J, L, lowP)) in enumerate(zip(walks, pre)):
    n, H, r = len(T), tr["H"], tr["root"]
    dy0 = walk_dy[tr["base"]:tr["base"] + n]
    # the ground: the cat's own lowest point (always a paw on the walks), smoothed over 0.5 s, never below the floor
    ground = lowP + r[:, 1] + dy0                      # room y (down)
    near = np.abs(ground - pg.floor) < FLOOR_NEAR
    ground = np.minimum(smooth(ground), pg.floor)
    st["on_furniture_frames"] += int((~near).sum())
    # lowering: with each paw at its standing place on the ground, how far down the body must go for EXT
    G = to_room(J[:, TOPS], H, r, dy0)
    Lr = to_room(L, H, r, dy0)
    Wr = to_room(J[:, WRISTS], H, r, dy0)
    WL = Wr - Lr
    N = nominal(G, H, ground)
    dv = N + WL - G
    full = np.linalg.norm(J[:, MIDS] - J[:, TOPS], axis=-1) + np.linalg.norm(J[:, WRISTS] - J[:, MIDS], axis=-1)
    tgt_len = EXT[None] * full
    hh2 = (dv[..., [0, 2]] ** 2).sum(-1)
    lower = dv[..., 1] - np.sqrt(np.maximum(tgt_len ** 2 - hh2, 0))
    st.setdefault("diag", []).append(np.c_[lower, ground[:, None] - G[..., 1], np.sqrt(np.maximum((STRETCH * full) ** 2 - hh2, 0)) + 0 * full,
                                           WL[..., 1], -WL[..., 1] * 0 + full])
    jf, jh = J[:, TOPS[:2]].mean(1), J[:, TOPS[2:]].mean(1)
    pivot = (jf + jh) / 2
    roll = np.zeros(n)
    if LEVEL == 2:  # the lowering, pitch and roll that best give each leg what it needs (least squares over the four)
        d = J[:, TOPS] - pivot[:, None]
        A = np.stack([np.ones((n, 4)), -d[..., 2], d[..., 0]], -1)  # down-shift = lower - pitch * dz + roll * dx
        sol = np.stack([np.linalg.lstsq(A[t], lower[t], rcond=None)[0] for t in range(n)])
        pitch = smooth(np.clip(sol[:, 1], -MAX_PITCH, MAX_PITCH))
        roll = smooth(np.clip(sol[:, 2], -MAX_PITCH, MAX_PITCH))
        lower = smooth(np.clip(sol[:, 0], -0.02, 0.05))
    elif LEVEL:  # the front and hind pairs' needs differ: pitch the body by the difference, lower it by the mean
        fneed, hneed = lower[:, :2].mean(1), lower[:, 2:].mean(1)
        pitch = smooth(np.clip((fneed - hneed) / np.linalg.norm(jf - jh, axis=1), -MAX_PITCH, MAX_PITCH))
        lower = smooth(np.clip((fneed + hneed) / 2, -0.02, 0.05))
    else:
        pitch = np.zeros(n)
        lower = smooth(np.clip(lower.min(1) if LOWER_BY == "min" else lower.mean(1), -0.02, 0.05))
    st["lowering_m"].append(lower)
    st.setdefault("pitch_deg", []).append(np.degrees(pitch))
    st.setdefault("roll_deg", []).append(np.degrees(roll))
    dy = dy0 + lower
    T = level(T, pitch, pivot, roll)
    # 30 steps a second (the page blends the recorded bones, then pitches them by the blended pitch and pivot)
    tk, rk, Hk, dyk, Tk_raw = interp(n, r, H, dy, tr["bones"].astype(np.float64))
    f = np.tile(np.arange(SUB) / SUB, n)
    t1 = np.minimum(tk + 1, n - 1)
    pk = pitch[tk] * (1 - f) + pitch[t1] * f
    ck = pivot[tk] * (1 - f)[:, None] + pivot[t1] * f[:, None]
    rk_ = roll[tk] * (1 - f) + roll[t1] * f
    Tk = level(Tk_raw, pk, ck, rk_)
    Jk = joints_of(Tk, Jr)
    # paw landmarks follow the bones linearly too (skinning is linear in the bones)
    Lk = level_pts(L[tk] * (1 - f)[:, None, None] + L[t1] * f[:, None, None], pk, ck, rk_)
    Gk = to_room(Jk[:, TOPS], Hk, rk, dyk)
    Wk = to_room(Jk[:, WRISTS], Hk, rk, dyk)
    WLk = Wk - to_room(Lk, Hk, rk, dyk)
    gk = ground[tk] * (1 - f) + ground[t1] * f
    Nk = nominal(Gk, Hk, gk)
    fwdk = axes(Hk)[1]
    fullk = full[tk] * (1 - f)[:, None] + full[t1] * f[:, None]
    K = len(Nk)
    mode, k0, since = np.zeros(4, int), np.zeros(4, int), np.full(4, MIN_STANCE)
    dur, last_lift = np.full(4, SWING_STEPS), np.full(4, -10 ** 9)
    lifts = [[] for _ in range(4)]
    F, S, Gt = Nk[0].copy(), Nk[0].copy(), Nk[0].copy()
    P = np.zeros((K, 4, 3))
    for k in range(K):
        for p in range(4):
            if mode[p] == 1 and k - k0[p] >= dur[p]:
                mode[p], F[p], since[p] = 0, Gt[p], 0
                F[p, 1] = gk[k] - DEPTH[p]
                st["swing_m"].append(float(np.hypot(*(Gt[p] - S[p])[[0, 2]])))
        e = np.hypot(*(F - Nk[k])[:, [0, 2]].T)
        far = np.linalg.norm(F + WLk[k] - Gk[k], axis=-1) > 0.999 * STRETCH * fullk[k]
        cand = [p for p in range(4) if mode[p] == 0 and since[p] >= MIN_STANCE and (e[p] >= TRIG or far[p])]
        cand.sort(key=lambda p: (not far[p], -e[p]))
        nsw = int(mode.sum())
        for p in cand:
            if nsw >= 2:
                break
            if mode[PARTNER[p]] == 1 and not far[p]:
                continue
            if SEQ and p < 2 and not far[p] and e[p] < 1.6 * TRIG and last_lift[p + 2] <= last_lift[p]:
                continue  # a front paw waits for its side's hind paw to step first
            d = SWING_STEPS
            if SWING_SPEED:
                kv = min(k + 9, K - 1)
                v = np.hypot(*(Nk[kv, p] - Nk[k, p])[[0, 2]]) / max((kv - k) / 30.0, 1e-6) / BODY_LEN
                d = int(round(30 * np.interp(v, [0.35, 0.65], [0.30, 0.20])))
            ka, kb = min(k + d, K - 1), min(k + d + LEAD_STEPS, K - 1)
            lead = (Nk[kb] - Nk[ka])[p] * np.array([1.0, 0.0, 1.0])
            ln = np.linalg.norm(lead)
            if ln > LEAD_CAP:
                lead *= LEAD_CAP / ln
            goal = Nk[ka, p] + lead
            if np.hypot(*(goal - F[p])[[0, 2]]) < MIN_STEP:
                continue
            if mode[PARTNER[p]] == 1:
                st["forced"] += 1
            st["stance_s"].append(since[p] / 30.0)
            mode[p], k0[p], S[p], Gt[p] = 1, k, F[p] + SLIDE * min(since[p] / SUB, 10.0) * fwdk[k], goal
            dur[p], last_lift[p] = d, k
            lifts[p].append(k)
            st.setdefault("swing_s", []).append(d / 30.0)
            st["steps"][p] += 1
            nsw += 1
        for p in range(4):
            if mode[p] == 0:
                P[k, p] = F[p] + SLIDE * min(since[p] / SUB, 10.0) * fwdk[k]
                since[p] += 1
                st["stance_steps"] += 1
                st["reach_limited_steps"] += int(far[p])
            else:
                sg = (k - k0[p]) / dur[p]
                sm = sg * sg * (3 - 2 * sg)
                P[k, p] = S[p] + (Gt[p] - S[p]) * sm
                lift = np.clip(LIFT_BASE + LIFT_PER_M * np.hypot(*(Gt[p] - S[p])[[0, 2]]), LIFT_MIN, LIFT_MAX)
                P[k, p, 1] = (1 - sg) * S[p, 1] + sg * (gk[k] - DEPTH[p]) - lift * np.sin(np.pi * sg)
    Wt = P + WLk                                        # wrist or hock targets, room
    # the body with the steps: the shoulders rise and fall (a lift of the front, as half a raise and a pitch) and the
    # head nods, by the real cats' waveforms over the stride of the -x hind paw; the paws stay where they are
    headk = np.zeros(K)
    if BODY_W is not None:
        phi, amp = stride_phase(lifts[2], K)
        if "shoulder" in BODY_W:
            sh = amp * wave(BODY_W["shoulder"], phi) * BODY_LEN          # m, + = shoulders higher
            Dg = np.linalg.norm(Jk[:, TOPS[:2]].mean(1) - Jk[:, TOPS[2:]].mean(1), axis=1)
            dyk = dyk - sh / 2
            pk = pk - sh / Dg
        if "head" in BODY_W:
            headk = amp * wave(BODY_W["head"], phi) * BODY_LEN / HEAD_R   # + nose down
        Tk = head_turn(level(Tk_raw, pk, ck, rk_), headk)
        st.setdefault("body_amp", []).append(amp)
        st.setdefault("head_deg", []).append(np.degrees(headk))
    WR.append((Wt - r[tk][:, None]).reshape(n, SUB, 4, 3))
    DYK.append(dyk.reshape(n, SUB))
    PK.append(pk.reshape(n, SUB))
    RL.append(rk_.reshape(n, SUB))
    CK.append(ck.reshape(n, SUB, 3))
    HK.append(headk.reshape(n, SUB))
    # the bent bones at the 0.1 s samples
    s0 = np.arange(0, K, SUB)
    dy = dyk[s0]
    T = head_turn(level(tr["bones"].astype(np.float64), pk[s0], ck[s0], rk_[s0]), headk[s0])
    tc = np.einsum("nji,npj->npi", H, Wt[s0] - r[:, None] - np.c_[np.zeros(n), dy, np.zeros(n)][:, None])
    Tb, _ = bend_to(T, Jr, tc)
    BONES.append(Tb.astype(np.float32))
    DY10.append(dy)
    if DENSE and tr.get("sub") == "free":               # every 30 Hz step, bent as the page bends it
        tck = np.einsum("kji,kpj->kpi", Hk, Wt - rk[:, None] - np.c_[np.zeros(K), dyk, np.zeros(K)][:, None])
        Tbk, _ = bend_to(Tk, Jr, tck)
        DN.append((tr["base"], n, Tbk.astype(np.float32), rk.astype(np.float32), Hk.astype(np.float32), dyk.astype(np.float32)))
    if i in (0, 5, 20, 60, 100) and K > 40:             # for the page's solver: a few steps, samples and between
        for k in rng.choice(K - SUB, 3, replace=False):
            tcat = Hk[k].T @ (Wt[k] - rk[k] - np.array([0, dyk[k], 0])).T
            Tb1, _ = bend_to(Tk[k:k + 1], Jr, tcat.T[None])
            tests.append({"run": i, "step": int(k), "chain_bones": CHAINS.tolist(),
                          "bones": Tb1[0][CHAINS.reshape(-1)].tolist(), "root": rk[k].tolist(), "dy": float(dyk[k]),
                          "H": Hk[k].tolist()})
    if i % 20 == 0:
        print("run %d/%d: %d frames, steps per paw so far %s" % (i + 1, len(walks), n, st["steps"]), flush=True)

WRa, DYa = np.concatenate(WR).astype(np.float16), np.concatenate(DYK).astype(np.float32)
assert np.abs(np.concatenate(WR)).max() < 60000
blobs = [("wrist", WRa), ("dy", DYa)]
if LEVEL or BODY:
    blobs += [("pitch", np.concatenate(PK).astype(np.float32)), ("pivot", np.concatenate(CK).astype(np.float32))]
if BODY:
    blobs += [("head", np.concatenate(HK).astype(np.float32))]
if LEVEL == 2:
    blobs += [("roll", np.concatenate(RL).astype(np.float32))]
meta, nbytes = pack(os.path.join(out, "gait.bin"), blobs)
json.dump({"blobs": meta, "bytes": nbytes, "sub": SUB, "chains": CHAINS.tolist(), "rest_joints": Jr.tolist(),
           "stretch": STRETCH, "pole": POLE.tolist(), "cut_deg": CUT_DEG, "max_bend_deg": MAX_BEND, "protect": list(PROTECT),
           "head_bone": HEAD_BONE, "head_joint": HEAD_JOINT,
           "units": "metres; wrist: wrist or hock target minus the cat's place at that 0.1 s frame, room axes (y down); "
                    "dy: room y shift per step; pitch (radians) and pivot (cat's frame): every bone turned about the cat's x "
                    "axis through the pivot before the legs bend", "from": "gait_cycle.py (the walking cycle on the copy's walks)"},
          open(os.path.join(out, "gait.json"), "w"), indent=1)
np.savez(os.path.join(out, "gait_bones.npz"), bones=np.concatenate(BONES), dy=np.concatenate(DY10))
if DENSE:  # per free walk: its first frame in the walks' data (base) and length (n, 0.1 s); arrays at 3 steps per frame
    np.savez(os.path.join(out, "gait_dense.npz"), base=np.array([d[0] for d in DN]), n=np.array([d[1] for d in DN]),
             bones=np.concatenate([d[2] for d in DN]), root=np.concatenate([d[3] for d in DN]),
             H=np.concatenate([d[4] for d in DN]), dy=np.concatenate([d[5] for d in DN]))
json.dump({"tests": tests}, open(os.path.join(out, "gait_test.json"), "w"))
low = np.concatenate(st["lowering_m"])
total_s = sum(len(tr["bones"]) for tr in walks) / 10
res = {"runs": len(walks), "seconds": total_s, "steps_per_paw": st["steps"],
       "steps_per_paw_per_minute": [round(s / total_s * 60, 1) for s in st["steps"]],
       "forced_steps_with_partner_in_air": st["forced"],
       "stance_s_q10_q50_q90": np.percentile(st["stance_s"], [10, 50, 90]).round(2).tolist() if st["stance_s"] else None,
       "swing_m_q10_q50_q90": np.percentile(st["swing_m"], [10, 50, 90]).round(3).tolist() if st["swing_m"] else None,
       "lowering_m_q10_q50_q90": np.percentile(low, [10, 50, 90]).round(4).tolist(),
       "pitch_deg_q10_q50_q90": np.percentile(np.concatenate(st["pitch_deg"]), [10, 50, 90]).round(2).tolist(),
       "roll_deg_q10_q50_q90": np.percentile(np.concatenate(st["roll_deg"]), [10, 50, 90]).round(2).tolist(),
       "body_motion_share_of_steps": float(np.mean(np.concatenate(st["body_amp"]) > 0.5)) if st.get("body_amp") else 0.0,
       "head_nod_deg_q5_q95": np.percentile(np.concatenate(st["head_deg"]), [5, 95]).round(2).tolist() if st.get("head_deg") else None,
       "reach_limited_share_of_stance_steps": st["reach_limited_steps"] / max(st["stance_steps"], 1),
       "frames_off_the_floor_share": st["on_furniture_frames"] / (total_s * 10),
       "standing_paw_from_top_joint": O_NOM.round(4).tolist(), "paw_landmark_above_sole": DEPTH.round(4).tolist(),
       "settings": {"EXT": EXT.tolist(), "LOWER_BY": LOWER_BY, "TRIG": TRIG, "SLANT": SLANT, "MAX_BEND": MAX_BEND, "swing_s": SWING_STEPS / 30, "lead_cap": LEAD_CAP,
                    "SWING_SPEED": SWING_SPEED, "SEQ": SEQ, "LIFT": [LIFT_MIN, LIFT_MAX], "LEVEL": LEVEL, "BODY": BODY, "BODY_SIZE": os.environ.get("BODY_SIZE", "mean")},
       "swing_time_s_q10_q50_q90": np.percentile(st.get("swing_s", [0.3]), [10, 50, 90]).round(3).tolist()}
D = np.concatenate(st["diag"])
res["per_leg_q10_q50_q90"] = {name: {"lowering_needed": np.percentile(D[:, p], [10, 50, 90]).round(3).tolist(),
                                     "top_joint_height": np.percentile(D[:, 4 + p], [10, 50, 90]).round(3).tolist(),
                                     "reach_down_to_wrist": np.percentile(D[:, 8 + p], [10, 50, 90]).round(3).tolist(),
                                     "wrist_above_paw": np.percentile(-D[:, 12 + p], [10, 50, 90]).round(3).tolist(),
                                     "chain_length": np.percentile(D[:, 16 + p], [10, 50, 90]).round(3).tolist()}
                              for p, name in enumerate(["front -x", "front +x", "hind -x", "hind +x"])}
json.dump(res, open(os.path.join(out, "gait_stats.json"), "w"), indent=1)
print(json.dumps(res, indent=1))
