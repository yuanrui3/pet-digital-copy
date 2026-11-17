"""Paws that stay planted, bent at the joints, legs that may lengthen (work/pets/PROTOCOL_CAT.md, "Paws that stay
planted, bent at the joints" and "... legs that may lengthen").

The plant-and-step state machine of feet_step.py, but each leg is bent through Pikachu's own skeleton instead of sliding
its surface. Joint j (1..25) sits at the origin of bone j-1; each leg is a chain of four joints (top, middle, wrist or
hock, paw). Per frame and leg with a correction: the top joint stays, the middle joint bends (keeping the leg's own bend
direction), the wrist/hock goes to where it was plus the paw's correction, the paw follows it without turning. The four
bones turn and move with their joints: top bone R1 (upper segment's turn), middle bone halfway between R1 and R2,
wrist/hock bone half of R2, paw bone moved only. A paw is held while its correction is at most 8 cm sideways and the
chain reaches it, lengthened by at most 5% (a straight leg cannot otherwise hold a paw while the hip glides on);
otherwise it steps (two frames in the air, 3 cm up, landing at its
natural place plus a lead capped at 4 cm). The grey cat follows Pikachu's bent bones through its usual retarget.
CPU only.

usage: python feet_ik.py PAGE_DIR CHECK_DIR UV_NPZ SKEL_NPZ OUT_DIR
Writes OUT_DIR/feet_ik.bin + .json (per recorded and walk frame and paw: the paw correction in the cat's own frame;
the chains' rest joints), feet_ik_check.json, feet_ik_test.json (bent bones on a few frames, to check the page's
solver), feet_ik_walk3.npz (bent bones of walk 3, 0-4 s, for feet_step_strip.py) and feet_ik.png.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feet_check import CHUNK, DOWN, FLOAT, PAWS, Page  # noqa: E402
from feet_fix import slide_share  # noqa: E402
from home2_retarget import region_retarget  # noqa: E402
from home2_transfer import edges_of, lbs, pack, read_blobs, stretch  # noqa: E402

CAP = 0.08        # sideways correction a leg may hold
STRETCH = 1.05    # a chain may lengthen by up to this factor
REACH = 0.999 * STRETCH  # share of the chain's full length it may reach to
LOWER = 0.02
LIFT = 0.03
LEAD_N, LEAD_CAP = 3, 0.04
EASE = (2 / 3, 1 / 3)
GIRDLE = (1, 0, 3, 2)
FREE, PLANTED, AIR = 0, 1, 2
# bones (top, middle, wrist/hock, paw) of each leg, in PAWS order: -x front, +x front, -x hind, +x hind
CHAINS = np.array([(8, 9, 10, 11), (4, 5, 6, 7), (21, 22, 23, 24), (17, 18, 19, 20)])


def rest_joints(page_dir, bones):
    """each bone's origin joint at rest, fitted to the recorded joints (paths.joints): R_b p + t_b = J_{b+1}"""
    pm = json.load(open(os.path.join(page_dir, "paths.json")))
    J = read_blobs(os.path.join(page_dir, "paths.bin"), pm)["joints"].reshape(-1, 25, 3).astype(np.float64)
    out, res = np.zeros((25, 3)), np.zeros(25)
    for b in range(25):
        R, t = bones[:, b, :, :3], bones[:, b, :, 3]
        y = (J[:, b] - t).reshape(-1)
        p, *_ = np.linalg.lstsq(R.reshape(-1, 3), y, rcond=None)
        out[b] = p
        res[b] = np.sqrt(((R.reshape(-1, 3) @ p - y) ** 2).reshape(-1, 3).sum(1).mean())
    return out, res


def joints_of(T, Jr):
    """T (n,25,3,4), Jr (25,3) -> each bone's origin joint in the pose (n,25,3)"""
    return np.einsum("nbij,bj->nbi", T[..., :3], Jr) + T[..., 3]


def rot_between(u, v):
    """(n,3) unit vectors -> (n,3,3) smallest rotation taking u to v"""
    ax = np.cross(u, v)
    c = np.clip((u * v).sum(-1), -0.999999, 1.0)
    K = np.zeros(u.shape[:-1] + (3, 3))
    K[..., 0, 1], K[..., 0, 2], K[..., 1, 2] = -ax[..., 2], ax[..., 1], -ax[..., 0]
    K[..., 1, 0], K[..., 2, 0], K[..., 2, 1] = ax[..., 2], -ax[..., 1], ax[..., 0]
    return np.eye(3) + K + (K @ K) / (1 + c)[..., None, None]


def half_turn(R):
    """(n,3,3) -> the rotation halfway from identity to R (same axis, half the angle)"""
    w = np.sqrt(np.clip(1 + R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2], 1e-12, None)) / 2
    x = (R[..., 2, 1] - R[..., 1, 2]) / (4 * w)
    y = (R[..., 0, 2] - R[..., 2, 0]) / (4 * w)
    z = (R[..., 1, 0] - R[..., 0, 1]) / (4 * w)
    q = np.stack([w + 1, x, y, z], -1)  # (1,0,0,0) + q, normalised: half the angle
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack([np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
                     np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
                     np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1)], -2)


def two_bone(Hp, Kp, Wp, tgt):
    """(n,3) top, middle, wrist joints and the wrist target -> new middle and wrist, R1, R2, lengthening factor.
    Beyond 99.9% of the chain's length both segments may lengthen by the same factor, up to STRETCH. The middle joint
    keeps the leg's own bend direction relative to the top-to-wrist line, turned with that line."""
    a = np.linalg.norm(Kp - Hp, axis=-1)
    b = np.linalg.norm(Wp - Kp, axis=-1)
    dv = tgt - Hp
    dn = np.maximum(np.linalg.norm(dv, axis=-1), 1e-9)
    u = dv / dn[..., None]
    s = np.clip(dn / (0.999 * (a + b)), 1.0, STRETCH)
    a2, b2 = a * s, b * s
    d = np.clip(dn, np.abs(a2 - b2) + 1e-3, 0.999 * (a2 + b2))
    w0 = Wp - Hp
    w = w0 / np.maximum(np.linalg.norm(w0, axis=-1), 1e-9)[..., None]
    pole0 = (Kp - Hp) - ((Kp - Hp) * w).sum(-1, keepdims=True) * w
    pole = np.einsum("nij,nj->ni", rot_between(w, u), pole0)
    old = (Kp - Hp) - ((Kp - Hp) * u).sum(-1, keepdims=True) * u  # the last try's rule, for straight legs
    pole = np.where((np.linalg.norm(pole0, axis=-1) < 1e-4)[..., None], old, pole)
    pole = pole - (pole * u).sum(-1, keepdims=True) * u
    v = pole / np.maximum(np.linalg.norm(pole, axis=-1), 1e-9)[..., None]
    cosA = np.clip((a2 * a2 + d * d - b2 * b2) / (2 * a2 * d), -1, 1)
    sinA = np.sqrt(1 - cosA ** 2)
    K2 = Hp + a2[..., None] * (cosA[..., None] * u + sinA[..., None] * v)
    W2 = Hp + d[..., None] * u
    R1 = rot_between((Kp - Hp) / a[..., None], (K2 - Hp) / a2[..., None])
    R2 = rot_between((Wp - Kp) / b[..., None], (W2 - K2) / b2[..., None])
    return K2, W2, R1, R2, s


def reachable(Hp, Kp, Wp, tgt):
    a, b = np.linalg.norm(Kp - Hp), np.linalg.norm(Wp - Kp)
    d = np.linalg.norm(tgt - Hp)
    return np.abs(a - b) + 0.005 <= d <= REACH * (a + b)


def bend(T, Jr, off, stretch_out=None):
    """T (n,25,3,4) bones, Jr rest joints, off (n,4,3) paw corrections (cat's own frame) -> bent bones;
    stretch_out (n,4), when given, gets each chain's lengthening factor"""
    T2 = T.copy()
    J = joints_of(T, Jr)
    for p, (bt, bm, bw, bp) in enumerate(CHAINS):
        m = np.abs(off[:, p]).max(1) > 1e-6
        if not m.any():
            continue
        Hp, Kp, Wp, Pp = J[m, bt], J[m, bm], J[m, bw], J[m, bp]
        K2, W2, R1, R2, s = two_bone(Hp, Kp, Wp, Wp + off[m, p])
        if stretch_out is not None:
            stretch_out[np.where(m)[0], p] = s
        Rm = R1 @ half_turn(np.swapaxes(R1, -1, -2) @ R2)
        for b, Rd, c, c2 in ((bt, R1, Hp, Hp), (bm, Rm, Kp, K2), (bw, half_turn(R2), Wp, W2), (bp, np.broadcast_to(np.eye(3), R1.shape), Pp, Pp + (W2 - Wp))):
            R, t = T[m, b, :, :3], T[m, b, :, 3]
            T2[m, b, :, :3] = Rd @ R
            T2[m, b, :, 3] = np.einsum("nij,nj->ni", Rd, t - c) + c2
    return T2


def steps(paw, low, down, h0, Hrot, Jt):
    """as feet_step.steps, but a paw is held while its sideways correction is at most CAP and its chain reaches it;
    Hrot (n,3,3) heading (world = H local + root), Jt (n,25,3) joints in the cat's frame"""
    n = len(paw)
    off = np.zeros((n, 4, 3))
    xz = paw[..., [0, 2]]
    state = np.full(4, FREE)
    anchor = np.zeros((4, 2))
    air_from, air_to, air_t0 = np.zeros((4, 2)), np.zeros((4, 2)), np.zeros(4, int)
    ease_left, ease_from = np.zeros(4, int), np.zeros((4, 3))
    planted_len = np.zeros(4, int)
    st = {"planted_runs": [], "run_ends": [], "steps": 0, "waits": 0, "planted_frames": 0, "reach_limited": 0}

    def lowered(t, p):
        return float(np.clip((low[t] - h0[p]) - paw[t, p, 1], 0.0, LOWER))

    def holds(t, p, o):
        if np.hypot(o[0], o[2]) > CAP:
            return False
        bt, bm, bw, _ = CHAINS[p]
        ok = reachable(Jt[t, bt], Jt[t, bm], Jt[t, bw], Jt[t, bw] + Hrot[t].T @ o)
        if not ok:
            st["reach_limited"] += 1
        return ok

    def end_plant(p, why):
        if planted_len[p]:
            st["planted_runs"].append(int(planted_len[p]))
            st["run_ends"].append(why)
        planted_len[p] = 0

    for t in range(n):
        in_air = [p for p in range(4) if state[p] == AIR and t - air_t0[p] < 2]
        for p in range(4):
            if state[p] == AIR and t - air_t0[p] >= 2:
                if down[t, p]:
                    state[p], anchor[p] = PLANTED, air_to[p]
                else:
                    state[p], ease_left[p] = FREE, len(EASE)
                    ease_from[p] = (air_to[p][0] - xz[t, p, 0], 0.0, air_to[p][1] - xz[t, p, 1])
        want = []
        for p in range(4):
            if state[p] == FREE:
                o = np.zeros(3)
                if ease_left[p] > 0:
                    o = EASE[len(EASE) - ease_left[p]] * ease_from[p]
                    ease_left[p] -= 1
                if down[t, p]:
                    state[p], anchor[p] = PLANTED, xz[t, p] + o[[0, 2]]
                else:
                    off[t, p] = o
            if state[p] == PLANTED:
                if not down[t, p]:
                    end_plant(p, "lift")
                    state[p], ease_left[p] = FREE, len(EASE) - 1
                    ease_from[p] = off[t - 1, p] if t else 0
                    off[t, p] = EASE[0] * ease_from[p]
                    continue
                d = anchor[p] - xz[t, p]
                o = np.array([d[0], lowered(t, p), d[1]])
                if not holds(t, p, o):
                    want.append((float(np.linalg.norm(d)), p))
                else:
                    off[t, p] = o
                    planted_len[p] += 1
                    st["planted_frames"] += 1
        for _, p in sorted(want, reverse=True):
            if len(in_air) < 2 and GIRDLE[p] not in in_air:
                end_plant(p, "step")
                prev = off[t - 1, p] if t else np.zeros(3)
                start = xz[t - 1, p] + prev[[0, 2]] if t else xz[t, p]
                tl = min(t + 2, n - 1)
                lead = xz[min(tl + LEAD_N, n - 1), p] - xz[tl, p]
                ln = np.linalg.norm(lead)
                if ln > LEAD_CAP:
                    lead *= LEAD_CAP / ln
                state[p], air_t0[p], air_from[p], air_to[p] = AIR, t, start, xz[tl, p] + lead
                in_air.append(p)
                st["steps"] += 1
            else:  # waits: the largest share of the correction it can hold, then slides with the body
                d = anchor[p] - xz[t, p]
                for s in np.linspace(1, 0, 11)[1:]:
                    o = np.array([s * d[0], lowered(t, p), s * d[1]])
                    if np.hypot(o[0], o[2]) <= CAP and reachable(*(Jt[t, CHAINS[p][k]] for k in range(3)), Jt[t, CHAINS[p][2]] + Hrot[t].T @ o):
                        break
                anchor[p] = xz[t, p] + s * d
                off[t, p] = o
                planted_len[p] += 1
                st["planted_frames"] += 1
                st["waits"] += 1
        for p in range(4):
            if state[p] == AIR and t - air_t0[p] < 2:
                f = (t - air_t0[p] + 1) / 3
                place = air_from[p] + f * (air_to[p] - air_from[p])
                up = min(0.0, (low[t] - h0[p] - LIFT) - paw[t, p, 1])
                off[t, p] = (place[0] - xz[t, p, 0], up, place[1] - xz[t, p, 1])
    for p in range(4):
        end_plant(p, "end")
    return off, st


def natural_runs(down, track, kind, k):
    out = []
    starts = np.r_[0, np.where(np.diff(track) != 0)[0] + 1, len(track)]
    for a, b in zip(starts[:-1], starts[1:]):
        if kind[a] != k:
            continue
        for p in range(4):
            d = np.diff(np.r_[0, down[a:b, p].astype(np.int8), 0])
            out += list(np.where(d == -1)[0] - np.where(d == 1)[0])
    return np.array(out)


def main():
    page_dir, check_dir, uv_npz, skel_npz, out = sys.argv[1:6]
    os.makedirs(out, exist_ok=True)
    page = Page(page_dir)
    F = np.load(os.path.join(check_dir, "feet_frames.npz"))
    track, kind, low = F["track"], F["kind"], F["lowW"]
    nrec = int((kind == "rec").sum())
    Tall = np.concatenate([tr["bones"] for tr in page.tracks]).astype(np.float64)
    Hall = np.concatenate([tr["H"] for tr in page.tracks])
    Rall = np.concatenate([tr["root"] for tr in page.tracks])
    assert len(Tall) == len(track)
    Jr, res = rest_joints(page_dir, Tall[:nrec])
    print("rest joints fitted; residual over the chain bones (mm): max %.2f" % (1000 * res[CHAINS.ravel()].max()), flush=True)
    Jt = joints_of(Tall, Jr)
    downP = F["hP"] <= DOWN
    h0 = np.array([page.P[s, 1].max() - page.P[s, 1].mean() for s in page.lp[:4]])
    starts = np.r_[0, np.where(np.diff(track) != 0)[0] + 1, len(track)]
    off = np.zeros_like(F["pawP"])
    st = {k: {"planted_runs": [], "run_ends": [], "steps": 0, "waits": 0, "planted_frames": 0, "reach_limited": 0} for k in ("rec", "walk")}
    for a, b in zip(starts[:-1], starts[1:]):
        oo, s = steps(F["pawP"][a:b], low[a:b], downP[a:b], h0, Hall[a:b], Jt[a:b])
        off[a:b] = oo
        for key in ("steps", "waits", "planted_frames", "reach_limited"):
            st[kind[a]][key] += s[key]
        st[kind[a]]["planted_runs"] += s["planted_runs"]
        st[kind[a]]["run_ends"] += s["run_ends"]
    print("state machine done", flush=True)
    loc = np.einsum("nji,npj->npi", Hall, off)  # cat's own frame: H^T off
    loc16 = loc.astype(np.float16)  # what the page gets; everything below uses it
    locq = loc16.astype(np.float64)

    # ---- re-pose every frame with the bent bones; paw places from the posed meshes
    pawP2, pawG2, low2 = np.zeros_like(F["pawP"]), np.zeros_like(F["pawG"]), np.zeros(len(track))
    jmove = np.zeros(len(track))
    lengthen = np.ones((len(track), 4))
    for a, b in zip(starts[:-1], starts[1:]):
        for c in range(a, b, CHUNK):
            e = min(b, c + CHUNK)
            sf = np.ones((e - c, 4))
            T2 = bend(Tall[c:e], Jr, locq[c:e], sf)
            lengthen[c:e] = sf
            jmove[c:e] = np.linalg.norm(joints_of(T2, Jr)[:, CHAINS[:, :3].ravel()] - Jt[c:e, CHAINS[:, :3].ravel()], axis=-1).max(1)
            XP, XG, lowP = page.pose(T2, Hall[c:e])
            LP, _ = page.split(XP, page.e3p, page.lp)
            LG, _ = page.split(XG, page.e3g, page.lg)
            H, r = Hall[c:e], Rall[c:e]
            pawP2[c:e] = np.einsum("nij,nlj->nli", H, LP[:, :4]) + r[:, None]
            pawG2[c:e] = np.einsum("nij,nlj->nli", H, LG[:, :4]) + r[:, None]
            low2[c:e] = lowP + r[:, 1]
    hP2, hG2 = low2[:, None] - pawP2[..., 1], low2[:, None] - pawG2[..., 1]
    print("re-posed", flush=True)

    lm = lengthen > 1 + 1e-6
    res_ = {"stretch_allowed": STRETCH, "chains_lengthened_share_of_paw_frames": float(lm.mean()),
            "lengthening_pct_median_p99_where_lengthened": [100 * (float(np.percentile(lengthen[lm], q)) - 1) for q in (50, 99)] if lm.any() else [],
            "cap_m": CAP, "reach": REACH, "lower_cap_m": LOWER, "lift_m": LIFT, "lead_cap_m": LEAD_CAP,
            "joint_move_cm_p50_p99_where_bent": [100 * float(np.percentile(jmove[jmove > 0], q)) for q in (50, 99)]}
    for cat, paw, h, paw2, h2 in (("pikachu", F["pawP"], F["hP"], pawP2, hP2), ("grey", F["pawG"], F["hG"], pawG2, hG2)):
        achieved = np.linalg.norm((paw2 - paw)[..., [0, 2]] - off[..., [0, 2]], axis=-1)
        for k in ("rec", "walk"):
            b_, nb, sb = slide_share(paw, h, track, kind, k)
            a_, na, sa = slide_share(paw2, h2, track, kind, k)
            s = st[k]
            runs, ends = np.array(s["planted_runs"]), np.array(s["run_ends"])
            down_s = float(downP[kind == k].sum()) / 10
            m = (kind == k)[:, None] & downP
            res_["%s %s" % (cat, k)] = {
                "skating_1.5_to_3cm_before": sb, "skating_1.5_to_3cm_after": sa,
                "skating_readout_after": "planted" if sa < 0.05 else "still skates",
                "M1_over_1.5cm_before": b_, "M1_over_1.5cm_after": a_, "down_steps_before": nb, "down_steps_after": na,
                "added_steps_per_paw_second_down": s["steps"] / max(down_s, 1e-9),
                "planted_stretch_s_median": float(np.median(runs)) / 10,
                "planted_stretch_s_p10_p90": [float(np.percentile(runs, q)) / 10 for q in (10, 90)],
                "planted_stretch_by_end": {w: {"share": float((ends == w).mean()), "median_s": float(np.median(runs[ends == w])) / 10}
                                           for w in ("step", "lift", "end") if (ends == w).any()},
                "motion_down_stretch_s_median": float(np.median(natural_runs(downP, track, kind, k))) / 10,
                "tapping_readout": "no tapping" if np.median(runs) >= 3 else "tapping",
                "waits_share_of_planted_frames": s["waits"] / max(1, s["planted_frames"]),
                "holds_refused_by_reach": s["reach_limited"],
                "paw_off_target_cm_median_p90_down": [100 * float(np.percentile(achieved[m], q)) for q in (50, 90)]}
    for k in ("rec", "walk"):
        m = (kind == k)[:, None] & downP
        res_["grey floating %s" % k] = {"share_over_3cm_before": float((F["hG"][m] > FLOAT).mean()),
                                        "share_over_3cm_after": float((hG2[m & (hP2 <= DOWN)] > FLOAT).mean())}

    # ---- stretch on 80 frames where a joint moves over 2 cm
    sk, z = np.load(skel_npz), np.load(uv_npz)
    Qc, own = z["pos"].astype(np.float64), sk["own"].astype(np.float64)
    eP, eG = edges_of(page.fp), edges_of(page.fg)
    rng = np.random.default_rng(1)
    big = jmove > 0.02
    rows = np.r_[rng.choice(np.where((kind == "rec") & big)[0], 40, replace=False),
                 rng.choice(np.where((kind == "walk") & big)[0], 40, replace=False)]
    stP0, stP1, stG = [], [], []
    for i in np.sort(rows):
        T = Tall[i:i + 1]
        T2 = bend(T, Jr, locq[i:i + 1])
        dn = Hall[i][1]
        XP0 = lbs(page.P, page.ip, page.wp, T[0])
        XP1 = lbs(page.P, page.ip, page.wp, T2[0])
        XG = lbs(page.Q, page.ig, page.wg, region_retarget(T2, page.mix, page.C)[0])
        stP0.append(stretch(page.P, XP0, eP))
        stP1.append(stretch(page.P, XP1, eP))
        stG.append(stretch(page.Q, XG, eG))
    st_own = np.concatenate([stretch(Qc, lbs(Qc, page.ig, page.wg, own[f]), eG) for f in np.linspace(0, len(own) - 1, 40).round().astype(int)])
    stP0, stP1, stG = np.concatenate(stP0), np.concatenate(stP1), np.concatenate(stG)
    ratio = float(np.percentile(stG, 99) / np.percentile(st_own, 99))
    o0, o1 = float((stP0 > np.log(1.3)).mean()), float((stP1 > np.log(1.3)).mean())
    res_["stretch"] = {"frames": "80 sampled where a joint moves over 2 cm", "grey_p99_over_own_video": ratio,
                       "grey_readout": "moves cleanly" if ratio <= 1.5 else ("some stretching" if ratio <= 3 else "tears"),
                       "pikachu_edges_over_30pct_before": o0, "pikachu_edges_over_30pct_with_bend": o1,
                       "pikachu_readout": "fine" if o1 - o0 <= 0.01 else "stretches"}
    json.dump(res_, open(os.path.join(out, "feet_ik_check.json"), "w"), indent=1)
    print(json.dumps(res_, indent=1), flush=True)

    # ---- for the page: paw corrections (cat's own frame) and the chains' rest joints
    meta, nbytes = pack(os.path.join(out, "feet_ik.bin"), [("rec", loc16[:nrec]), ("walk", loc16[nrec:])])
    json.dump({"blobs": meta, "bytes": nbytes, "paws": PAWS, "chains": CHAINS.tolist(), "stretch": STRETCH, "rest_joints": np.round(Jr, 6).tolist(),
               "rec_frames": nrec, "walk_frames": int(len(track) - nrec), "units": "metres, the cat's own frame (y down)",
               "from": "paws planted, short lifted steps, legs bent at their joints (two-bone chains); feet_ik.py"},
              open(os.path.join(out, "feet_ik.json"), "w"))
    # a few frames' bent bones, to check the page's solver against
    pick = [int(i) for i in np.r_[np.where((kind == "rec") & big)[0][::4000][:3], np.where((kind == "walk") & big)[0][::15000][:3]]]
    test = []
    for i in pick:
        T2 = bend(Tall[i:i + 1], Jr, locq[i:i + 1])[0]
        test.append({"kind": str(kind[i]), "index": int(i if kind[i] == "rec" else i - nrec), "chain_bones": CHAINS.tolist(),
                     "bones": np.round(T2[CHAINS.ravel()], 6).tolist()})
    json.dump(test, open(os.path.join(out, "feet_ik_test.json"), "w"))
    w3 = [i for i, tr in enumerate(page.tracks) if tr["name"] == "walk 3"][0]
    m = np.where(track == w3)[0][:41]
    np.savez(os.path.join(out, "feet_ik_walk3.npz"), bones=bend(Tall[m], Jr, locq[m]), t=F["t"][m])
    print("feet_ik: %d bytes; test frames %s" % (nbytes, pick), flush=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    mm = (track == w3) & (F["t"] < 200)
    tt = F["t"][mm] / 10
    fig, axes = plt.subplots(4, 2, figsize=(16, 11), sharex=True)
    for p in range(4):
        for c, (nm, paw, paw2, l1, l2, col) in enumerate((("Pikachu", F["pawP"], pawP2, low, low2, "tab:orange"), ("grey cat", F["pawG"], pawG2, low, low2, "tab:blue"))):
            ax = axes[p, c]
            for X, L, ls, lab in ((paw, l1, ":", "before"), (paw2, l2, "-", "after")):
                ax.plot(tt, 100 * (L[mm] - X[mm, p, 1]), ls, c=col, label="height, " + lab)
                ax.plot(tt, 100 * np.r_[0, np.linalg.norm(np.diff(X[mm, p][:, [0, 2]], axis=0), axis=1)], ls, c="k", lw=0.7, label="sideways move per 0.1 s, " + lab)
            ax.axhspan(1.5, 3, color="red", alpha=0.08)
            ax.set_ylim(-1, 15)
            ax.set_ylabel("%s\n%s (cm)" % (nm, PAWS[p]))
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "feet_ik.png"), dpi=70)


if __name__ == "__main__":
    main()
