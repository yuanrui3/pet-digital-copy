"""Stand the grey cat square through a skeleton (work/pets/PROTOCOL_CAT.md, "Standing the grey cat square").

The grey cat's rest pose (its own video's frame 50) has its +x front paw about 8 cm up and its hind paws one behind the
other. Its own learnt skeleton cannot fix that: one bone ("L hand") carries both front legs, and its joints are not at
elbows or wrists. So each leg gets a simple skeleton of its own, from its shape: its top (where it meets the body), one
joint at its strongest bend, and the paw tip; two bones (upper, lower) whose weights fade over 1.5 cm at the joint and
at the top. Each leg is bent at those joints (two-joint reach) so each paw stands straight below its leg's top, each pair
side by side at least 5 cm apart, all four at the height of the lowest paw. The mesh's UV seams are welded first, so
distances along the surface cross them (unwelded, each paw was cut into separate pieces).

That first try (SQUARE=legs) failed: the grey cat's model has one fused front leg, with a bump where the second paw
should be. SQUARE=hind_own (used next): only the two hind legs, which are separate and have bones of their own, are bent
through the grey cat's own skeleton: "R ankle"/"R foot" (-x hind) and "L ankle"/"L foot" (+x hind), each upper bone
turning about its own joint and the lower about the hock, the mesh following by its own skinning from its canonical
shape. Targets: each hind paw below its upper joint (pair side by side, at least 5 cm apart, level with the lowest paw);
if that fails the squaring check, the pair squared where it stands on average.

Then, SPLIT=1 (first try only): the +x front leg also gets its own bone for motion (its leg weight taken from the bones it
shared with the -x front leg), so each front leg follows the part of Pikachu it corresponds to. SPLIT=0: the 25 bones as
now. Both redo the correspondence and bone mixes from the squared rest, as home2_retarget.py does (MODE=region).
CPU only.

usage: python home2_stand.py PAGE_DIR UV_NPZ SKEL_NPZ OUT_DIR
  PAGE_DIR: the page's data (a2s_cat = Pikachu, a2s_cat2 = the grey cat as now, a2s_walks)
  UV_NPZ, SKEL_NPZ: the grey cat's canonical mesh and its own skeleton and motion (for its own video's stretch)
Writes OUT_DIR/a2s_cat2.bin + .json (the squared rest, skinning and mixes, for the page), home2_stand.json and
home2_stand.png.
"""
import json
import os
import sys

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from home2_retarget import NAMES, landmarks, nonrigid, region_mix, region_retarget, rot_between  # noqa: E402
from feet_check import landmark_sets  # noqa: E402
from home2_transfer import edges_of, lbs, pack, read_blobs, stretch, umeyama  # noqa: E402

PAWS = ["paw -x front", "paw +x front", "paw -x hind", "paw +x hind"]
FADE = 0.015    # weights fade over this on each side of the joint and of the leg's top
APART = 0.05    # each pair at least this far apart sideways
RING = 0.005    # half-width of a ring of equal distance from the paw tip
CHECK_FRAMES = 40


def weld(X, faces):
    """vertices at the same place joined (the UV seams) -> welded index per vertex, welded positions, welded edges"""
    key = np.round(X * 1e6).astype(np.int64)
    _, first, wid = np.unique(key, axis=0, return_index=True, return_inverse=True)
    wid = wid.ravel()
    e = edges_of(wid[faces])
    return wid, X[first], e[e[:, 0] != e[:, 1]]


def geodesic(Xw, e, src):
    L = np.linalg.norm(Xw[e[:, 0]] - Xw[e[:, 1]], axis=1)
    G = coo_matrix((np.r_[L, L], (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])), shape=(len(Xw),) * 2).tocsr()
    return dijkstra(G, indices=src)


def smooth_step(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def leg_skeleton(Xw, d, tip):
    """a leg's top (where the ring of equal distance from the paw tip suddenly widens: the leg meets the body), its
    centre line, and one joint at its strongest bend (the middle if it is straight)"""
    steps = np.arange(0.01, 0.40, 0.005)
    cen, spread = [], []
    for s in steps:
        m = np.abs(d - s) < RING
        if m.sum() < 5:
            cen.append(np.full(3, np.nan))
            spread.append(np.nan)
            continue
        c = Xw[m].mean(0)
        cen.append(c)
        spread.append(np.percentile(np.linalg.norm(Xw[m] - c, axis=1), 90))
    cen, spread = np.array(cen), np.array(spread)
    base = np.nanmedian(spread[(steps >= 0.02) & (steps < 0.06)])
    wide = np.where((steps >= 0.06) & (spread > 2 * base))[0]
    k_top = int(wide[0]) if len(wide) else int(np.searchsorted(steps, 0.16))
    L = float(steps[k_top])
    top = cen[max(k_top - 2, 0)]  # 1 cm below where it widens: still the leg's own centre
    cand = np.where((steps >= 0.03) & (steps <= L - 0.03))[0]
    axis = (top - tip) / np.linalg.norm(top - tip)
    dev = np.array([np.linalg.norm((cen[k] - tip) - ((cen[k] - tip) @ axis) * axis) for k in cand])
    if len(cand) and np.nanmax(dev) >= 0.01:
        kj = int(cand[np.nanargmax(dev)])
    else:
        kj = int(np.searchsorted(steps, L / 2))
    return {"length": L, "top": top, "joint": cen[kj], "joint_d": float(steps[kj]), "bend_cm": 100 * float(np.nanmax(dev)) if len(cand) else 0.0,
            "profile": [(round(100 * float(s), 1), round(100 * float(p), 2)) for s, p in zip(steps, spread)][:60], "ring_base_cm": 100 * float(base)}


def reach(T, J, E, tgt):
    """two-joint reach: upper bone turns about the top T, lower about the joint J, the bend kept in the leg's own plane
    (and on the same side). -> upper rotation Ru, lower rotation Rl (applied after Ru), new joint J2, where the tip ends"""
    l1, l2 = np.linalg.norm(J - T), np.linalg.norm(E - J)
    D = np.linalg.norm(tgt - T)
    u = (tgt - T) / D
    n = np.cross(J - T, E - J)
    if np.linalg.norm(n) < 1e-6:  # straight: bend forward/back, about the sideways axis
        n = np.cross(u, [1.0, 0, 0])
    n /= np.linalg.norm(n)
    # the side of the line T->E the joint was on, carried into the new plane
    w = np.cross(n, u)
    w /= np.linalg.norm(w)
    old_side = (J - T) - ((J - T) @ ((E - T) / np.linalg.norm(E - T))) * (E - T) / np.linalg.norm(E - T)
    if old_side @ w < 0:
        w = -w
    if D >= l1 + l2 - 1e-9:
        J2, reached = T + l1 * u, False
    else:
        ca = np.clip((l1 * l1 + D * D - l2 * l2) / (2 * l1 * D), -1, 1)
        J2, reached = T + l1 * (ca * u + np.sqrt(1 - ca * ca) * w), True
    Ru = rot_between(J - T, J2 - T)
    Rl = rot_between(Ru @ (E - J), (tgt - J2) if reached else u)
    E2 = J2 + Rl @ Ru @ (E - J)
    return Ru, Rl, J2, E2, reached


HIND_OWN = {2: (23, 24), 3: (19, 20)}  # paw index -> (upper, lower) bone of the grey cat's own skeleton


def rigid_about(Rm, src, dst):
    """3x4: rotate by Rm about src and carry src to dst"""
    return np.c_[Rm, dst - Rm @ src]


def own_pose(Qc, ic, wc, A, M):
    """the grey cat posed from its canonical mesh by its own skinning: each bone b by A[b], then M[b] (3x4 each)"""
    T = np.zeros_like(A)
    T[:, :, :3] = M[:, :, :3] @ A[:, :, :3]
    T[:, :, 3] = np.einsum("bij,bj->bi", M[:, :, :3], A[:, :, 3]) + M[:, :, 3]
    return lbs(Qc, ic, wc, T)


def squaring_check(Qa, Qs, fg, moved, tgt, which):
    """share of the moved part's edges stretched by over 30%, and how far the paw search on the squared mesh finds
    each paw in `which` from its target"""
    e = edges_of(fg)
    st = stretch(Qa, Qs, e)
    ok_e = np.linalg.norm(Qa[e[:, 0]] - Qa[e[:, 1]], axis=1) > 1e-9
    leg_e = (moved[e[:, 0]] | moved[e[:, 1]])[ok_e]
    over30 = float((st[leg_e] > np.log(1.3)).mean())
    try:
        lm2 = landmarks(Qs, fg)
        found = np.stack([lm2[p] for p in PAWS])
        off = 100 * np.linalg.norm(found - tgt, axis=1)
    except AssertionError:
        found, off = None, np.full(4, np.inf)
    ok = over30 <= 0.02 and bool(np.all(off[which] <= 2.0))
    return over30, off, ok, found


def square_hind_own(Qa, fg, uv_npz, skel_npz):
    """bend the two hind legs through the grey cat's own skeleton (see the module's docstring)"""
    z = np.load(uv_npz)
    skd = np.load(skel_npz)
    Qc, own = z["pos"].astype(np.float64), skd["own"].astype(np.float64)
    ic, wc = skd["skin_idx"].astype(np.int64), skd["skin_w"].astype(np.float64)
    f0 = int(os.environ.get("REBASE_FRAME", "50"))
    X0 = lbs(Qc, ic, wc, own[f0])
    s_, R_, t_ = umeyama(X0, Qa)
    A = own[f0].copy()
    A[:, :, :3] = s_ * R_ @ own[f0][:, :, :3]
    A[:, :, 3] = s_ * own[f0][:, :, 3] @ R_.T + t_
    B = len(A)
    I = np.tile(np.eye(3, 4), (B, 1, 1))
    resid = np.abs(own_pose(Qc, ic, wc, A, I) - Qa).max()
    print("own frame %d posed from the canonical mesh matches the page's rest to %.1e m" % (f0, resid), flush=True)
    assert resid < 1e-4
    J = np.einsum("bij,bj->bi", A[:, :, :3], skd["joints"].astype(np.float64)) + A[:, :, 3]
    sets = landmark_sets(Qa, fg)
    tips = np.stack([Qa[sets[k]].mean(0) for k in range(4)])
    Wd = np.zeros((len(Qa), B))
    np.add.at(Wd, (np.arange(len(Qa))[:, None], ic), wc)
    moved_bones = [b for ub_lb in HIND_OWN.values() for b in ub_lb]
    moved = Wd[:, moved_bones].sum(1) > 0.05
    for k, (ub, lb) in HIND_OWN.items():
        print("%s: own upper bone %s, joint %s; lower bone %s, joint %s; paw %s (cm)" % (
            PAWS[k], NAMES[ub], np.round(100 * J[ub], 1), NAMES[lb], np.round(100 * J[lb], 1), np.round(100 * tips[k], 1)), flush=True)
    lm0 = landmarks(Qa, fg)  # the paw search as it finds the front paws now: they must stay there
    front0 = np.stack([lm0[p] for p in PAWS[:2]])
    level = tips[:, 1].max()
    tries = []
    for kind in ("below the top", "in place"):
        tgt = tips.copy()  # front paws stay
        for k, (ub, _) in HIND_OWN.items():
            tgt[k] = J[ub] if kind == "below the top" else tips[[2, 3]].mean(0)
        tgt[[2, 3], 1] = level
        tgt[[2, 3], 2] = tgt[[2, 3], 2].mean()
        if kind == "in place" or tgt[3, 0] - tgt[2, 0] < APART:
            mid = tgt[[2, 3], 0].mean()
            tgt[2, 0], tgt[3, 0] = mid - APART / 2, mid + APART / 2
        print("targets, %s (cm): " % kind + "; ".join("%s %s (move %s)" % (PAWS[k], np.round(100 * tgt[k], 1), np.round(
            100 * (tgt[k] - tips[k]), 1)) for k in HIND_OWN), flush=True)
        adj, moves = tgt.copy(), {}
        for it in range(5):
            M = I.copy()
            for k, (ub, lb) in HIND_OWN.items():
                Ru, Rl, J2, E2, ok = reach(J[ub], J[lb], tips[k], adj[k])
                M[ub] = rigid_about(Ru, J[ub], J[ub])
                M[lb] = rigid_about(Rl @ Ru, J[lb], J2)
                ang = lambda Rm: float(np.degrees(np.arccos(np.clip((np.trace(Rm) - 1) / 2, -1, 1))))  # noqa: E731
                moves[PAWS[k]] = {"upper_deg": ang(Ru), "lower_deg": ang(Rl), "reached": bool(ok)}
            Qs = own_pose(Qc, ic, wc, A, M)
            got = np.stack([Qs[sets[k]].mean(0) for k in range(4)])
            err = tgt - got
            print("  pass %d: paws land %s cm from their targets" % (it + 1, np.round(100 * np.linalg.norm(err[[2, 3]], axis=1), 2)), flush=True)
            if np.abs(err[[2, 3]]).max() < 0.002:
                break
            adj[[2, 3]] += err[[2, 3]]
        for k in HIND_OWN:
            moves[PAWS[k]]["landed_off_target_cm"] = 100 * float(np.linalg.norm(err[k]))
            print("  %s: upper bone turns %.0f deg, lower %.0f deg" % (PAWS[k], moves[PAWS[k]]["upper_deg"], moves[PAWS[k]]["lower_deg"]), flush=True)
        tgt[[0, 1]] = front0
        chk = squaring_check(Qa, Qs, fg, moved, tgt, [0, 1, 2, 3])
        print("squaring (%s): %.2f%% of the moved legs' edges stretched over 30%% (limit 2%%); paws found %s cm from their "
              "targets (limit 2): %s" % (kind, 100 * chk[0], np.round(chk[1], 1), "PASSES" if chk[2] else "FAILS"), flush=True)
        tries.append({"targets": kind, "tgt": np.round(tgt, 4).tolist(), "moves": moves, "edges_over_30pct": chk[0],
                      "paws_found_off_target_cm": np.round(chk[1], 2).tolist(), "passes": chk[2]})
        if chk[2]:
            break
    chains = [np.stack([J[ub], J[lb], tips[k]]) for k, (ub, lb) in HIND_OWN.items()]
    return Qs, tips, tgt, moved, chains, chk, tries


def main():
    page_dir, uv_npz, skel_npz, out = sys.argv[1:5]
    split = os.environ.get("SPLIT", "1") == "1"
    os.makedirs(out, exist_ok=True)

    def blobs(name):
        m = json.load(open(os.path.join(page_dir, name + ".json")))
        return m, read_blobs(os.path.join(page_dir, name + ".bin"), m)
    am, A = blobs("a2s_cat")
    gm, G = blobs("a2s_cat2")
    wm, W = blobs("a2s_walks")
    B = am["bones"]
    P, fp = A["pos"].astype(np.float64), A["faces"].astype(np.int64)
    ip, wp = A["skin_idx"].astype(np.int64), A["skin_w"].astype(np.float64)
    Qa, fg, uv = G["pos"].astype(np.float64), G["faces"].astype(np.int64), G["uv"].astype(np.float32)
    ig, wg = G["skin_idx"].astype(np.int64), G["skin_w"].astype(np.float64)
    rec = A["bones"].astype(np.float64).reshape(-1, B, 3, 4)
    walks = W["bones"].astype(np.float64).reshape(-1, B, 3, 4)
    V = len(Qa)

    mode = os.environ.get("SQUARE", "legs")
    if mode == "legs":  # first try: a simple skeleton per leg, from its shape
        # ---- weld the seams; each leg's own skeleton
        wid, Xw, ew = weld(Qa, fg)
        print("welded: %d vertices -> %d (%d seam copies); %d edges" % (V, len(Xw), V - len(Xw), len(ew)), flush=True)
        lm = landmarks(Qa, fg)
        tips = np.stack([lm[p] for p in PAWS])
        dist, legs = [], []
        for k, p in enumerate(PAWS):
            src = int(np.argmin(np.linalg.norm(Xw - tips[k], axis=1)))
            dw = geodesic(Xw, ew, src)
            dist.append(dw[wid])
            sk = leg_skeleton(Xw, dw, tips[k])
            legs.append(sk)
            print("%s: tip %s cm; leg %.1f cm along the surface (ring %.1f cm wide below), joint at %.1f cm (bend %.1f cm off "
                  "straight), top %s, joint %s" % (p, np.round(100 * tips[k], 1), 100 * sk["length"], sk["ring_base_cm"],
                                                    100 * sk["joint_d"], sk["bend_cm"], np.round(100 * sk["top"], 1),
                                                    np.round(100 * sk["joint"], 1)), flush=True)
        others = [float(dist[a][np.argmin(np.linalg.norm(Qa - tips[b], axis=1))]) for a, b in ((0, 1), (2, 3), (0, 2))]
        print("along the surface between paws (cm): front pair %.1f, hind pair %.1f, -x front to -x hind %.1f" % tuple(
            100 * np.array(others)), flush=True)

        # ---- targets: each paw below its leg's top; pairs side by side, at least APART apart; level at the lowest paw
        tops = np.stack([sk["top"] for sk in legs])
        tgt = tops.copy()
        tgt[:, 1] = tips[:, 1].max()
        for a, b in ((0, 1), (2, 3)):  # (-x, +x) of each pair
            tgt[[a, b], 2] = tgt[[a, b], 2].mean()
            if tgt[b, 0] - tgt[a, 0] < APART:
                mid = tgt[[a, b], 0].mean()
                tgt[a, 0], tgt[b, 0] = mid - APART / 2, mid + APART / 2
        print("targets (cm): " + "; ".join("%s %s (move %s)" % (p, np.round(100 * t, 1), np.round(100 * (t - e), 1))
                                           for p, t, e in zip(PAWS, tgt, tips)), flush=True)

        # ---- per-leg bones and weights; bend
        lower, legw = np.zeros((4, V)), np.zeros((4, V))
        for k, sk in enumerate(legs):
            d = dist[k]
            lower[k] = 1 - smooth_step((d - (sk["joint_d"] - FADE)) / (2 * FADE))
            legw[k] = 1 - smooth_step((d - (sk["length"] - FADE)) / (2 * FADE))
        tot = legw.sum(0)
        over = tot > 1
        legw[:, over] /= tot[over]
        lower = np.minimum(lower, legw)
        print("vertices on legs: %s; on two legs at once: %d" % ([int((legw[k] > 0.5).sum()) for k in range(4)], int(over.sum())),
              flush=True)
        Qs = (1 - legw.sum(0))[:, None] * Qa
        reached, ends, moves = [], [], []
        for k, sk in enumerate(legs):
            T, J, E = sk["top"], sk["joint"], tips[k]
            Ru, Rl, J2, E2, ok = reach(T, J, E, tgt[k])
            up = T + (Qa - T) @ Ru.T
            lo = J2 + (Qa - J) @ (Rl @ Ru).T
            Qs += (legw[k] - lower[k])[:, None] * up + lower[k][:, None] * lo
            reached.append(bool(ok))
            ends.append(E2)
            ang_u = np.degrees(np.arccos(np.clip((np.trace(Ru) - 1) / 2, -1, 1)))
            ang_l = np.degrees(np.arccos(np.clip((np.trace(Rl) - 1) / 2, -1, 1)))
            moves.append({"upper_deg": float(ang_u), "lower_deg": float(ang_l), "reached": bool(ok),
                          "tip_end_off_target_cm": 100 * float(np.linalg.norm(E2 - tgt[k]))})
            print("  %s: upper turns %.0f deg, lower %.0f deg; %s (tip %.1f cm from target)" % (
                PAWS[k], ang_u, ang_l, "reached" if ok else "out of reach, straightened", 100 * np.linalg.norm(E2 - tgt[k])), flush=True)

        # ---- squaring checks: stretch on leg edges; the paw search on the squared mesh
        e = edges_of(fg)
        st = stretch(Qa, Qs, e)
        on_leg = legw.sum(0) > 0
        ok_e = np.linalg.norm(Qa[e[:, 0]] - Qa[e[:, 1]], axis=1) > 1e-9
        leg_e = (on_leg[e[:, 0]] | on_leg[e[:, 1]])[ok_e]
        over30 = float((st[leg_e] > np.log(1.3)).mean())
        try:
            lm2 = landmarks(Qs, fg)
            found = np.stack([lm2[p] for p in PAWS])
            off = 100 * np.linalg.norm(found - tgt, axis=1)
        except AssertionError:
            found, off = None, np.full(4, np.inf)
        sq_ok = over30 <= 0.02 and bool(np.all(off <= 2.0))
        print("squaring: %.2f%% of leg edges stretched over 30%% (limit 2%%); paws found %s cm from their targets (limit 2): %s" % (
            100 * over30, np.round(off, 1), "PASSES" if sq_ok else "FAILS"), flush=True)
        if found is not None:
            sq = found
            print("square: level within %.1f cm; pairs %.1f and %.1f cm apart sideways, %.1f and %.1f cm front to back" % (
                100 * (sq[:, 1].max() - sq[:, 1].min()), 100 * (sq[1, 0] - sq[0, 0]), 100 * (sq[3, 0] - sq[2, 0]),
                100 * abs(sq[1, 2] - sq[0, 2]), 100 * abs(sq[3, 2] - sq[2, 2])), flush=True)

        chains = [np.stack([sk["top"], sk["joint"], tips[k]]) for k, sk in enumerate(legs)]
        moved, tries = legw.sum(0) > 0, None
    else:  # SQUARE=hind_own: the hind legs through the grey cat's own skeleton
        assert not split, "SPLIT=1 belongs to the first try"
        Qs, tips, tgt, moved, chains, chk, tries = square_hind_own(Qa, fg, uv_npz, skel_npz)
        over30, off, sq_ok, found = chk
        moves, legs = tries, []
        if found is not None:
            sq = found
            print("square: level within %.1f cm; hind pair %.1f cm apart sideways, %.1f cm front to back" % (
                100 * (sq[:, 1].max() - sq[:, 1].min()), 100 * (sq[3, 0] - sq[2, 0]), 100 * abs(sq[3, 2] - sq[2, 2])), flush=True)

    e = edges_of(fg)
    if not sq_ok:
        json.dump({"square": mode, "tries": tries, "squaring_leg_edges_over_30pct": over30,
                   "paws_found_off_target_cm": np.round(off, 2).tolist(), "squaring_passes": False},
                  open(os.path.join(out, "home2_stand.json"), "w"), indent=1)
        picture(Qa, Qs, moved, chains, tgt, os.path.join(out, "home2_stand.png"))
        print("NOT KEPT: the squaring fails its own check", flush=True)
        return

    # ---- skinning for motion: SPLIT=1 gives the +x front leg its own bone (index B)
    if split:
        f = legw[1]
        Wd = np.zeros((V, B + 1))
        np.add.at(Wd, (np.arange(V)[:, None], ig), wg * (1 - f)[:, None])
        Wd[:, B] = f
        i4 = np.argsort(-Wd, 1)[:, :4]
        w4 = np.take_along_axis(Wd, i4, 1)
        cover = w4.sum(1)
        w4 /= cover[:, None]
        print("split: +x front leg's own bone %d on %d vertices (weight > 0.5); four strongest bones cover %.4f on average "
              "(worst %.3f)" % (B, int((f > 0.5).sum()), cover.mean(), cover.min()), flush=True)
    else:
        i4, w4 = ig, wg

    # ---- correspondence and bone mixes from the squared rest (as home2_retarget.py, MODE=region)
    tree_p = cKDTree(P)
    X = nonrigid(Qs, fg, P, fp, tree_p, [30.0, 10.0, 3.0])
    mix, C, share = region_mix(Qs, X, i4, w4, P, ip, wp, B)
    for i in range(len(mix)):
        if share[i] > 0.005:
            nm = NAMES[i] if i < B else "+x front leg"
            print("  grey %-12s (%4.1f%% of vertices) <- %s" % (nm, 100 * share[i], ", ".join(
                "%s %.0f%%" % (NAMES[a], 100 * b) for a, b in mix[i][:3])), flush=True)

    # ---- stretch against its own video (as home2_retarget.py)
    z = np.load(uv_npz)
    skd = np.load(skel_npz)
    Qc, own = z["pos"].astype(np.float64), skd["own"].astype(np.float64)
    ic, wc = skd["skin_idx"].astype(np.int64), skd["skin_w"].astype(np.float64)
    st_own = np.concatenate([stretch(Qc, lbs(Qc, ic, wc, own[f_]), e) for f_ in
                             np.linspace(0, len(own) - 1, min(CHECK_FRAMES, len(own))).round().astype(int)])
    fr = np.linspace(0, len(rec) - 1, CHECK_FRAMES).round().astype(int)
    fw = np.linspace(0, len(walks) - 1, CHECK_FRAMES).round().astype(int)
    st_g = []
    for T_ in (rec[fr], walks[fw]):
        Gm = region_retarget(T_, mix, C)
        for k in range(len(T_)):
            st_g.append(stretch(Qs, lbs(Qs, i4, w4, Gm[k]), e))
    st_g = np.concatenate(st_g)
    q_own, q_g = float(np.percentile(st_own, 99)), float(np.percentile(st_g, 99))
    ratio = q_g / q_own
    print("stretch: 99th percentile %.3f vs its own video %.3f: %.2f times -> %s; edges over 30%%: %.1f%% (own video %.1f%%)" % (
        q_g, q_own, ratio, "moves cleanly" if ratio <= 1.5 else ("some stretching" if ratio <= 3 else "tears"),
        100 * (st_g > np.log(1.3)).mean(), 100 * (st_own > np.log(1.3)).mean()), flush=True)

    res = {"square": mode, "split": split, "tries": tries, "legs": [{k: v for k, v in sk.items() if k not in ("top", "joint")} |
           {"top": np.round(sk["top"], 4).tolist(), "joint": np.round(sk["joint"], 4).tolist()} for sk in legs],
           "tips": np.round(tips, 4).tolist(), "targets": np.round(tgt, 4).tolist(), "moves": moves,
           "squaring_leg_edges_over_30pct": over30, "paws_found_off_target_cm": np.round(off, 2).tolist(), "squaring_passes": sq_ok,
           "stretch_p99_own": q_own, "stretch_p99_moved": q_g, "stretch_ratio": ratio,
           "stretch_readout": "moves cleanly" if ratio <= 1.5 else ("some stretching" if ratio <= 3 else "tears"),
           "bones": len(mix)}
    json.dump(res, open(os.path.join(out, "home2_stand.json"), "w"), indent=1)

    meta, nbytes = pack(os.path.join(out, "a2s_cat2.bin"), [
        ("faces", fg.astype(np.uint32)), ("pos", Qs.astype(np.float32)), ("uv", uv),
        ("skin_idx", i4.astype(np.uint8)), ("skin_w", w4.astype(np.float32))])
    gj = {k: v for k, v in gm.items() if k not in ("blobs", "bytes", "mix", "centre", "bones", "vertices")}
    gj.update({"blobs": meta, "bytes": nbytes, "vertices": V, "bones": len(mix), "mix": mix,
               "centre": np.round(C, 6).tolist(), "squared": "legs bent at a simple skeleton of their own to stand square "
                                                           "(home2_stand.py)" + (", +x front leg with its own bone" if split else "")})
    json.dump(gj, open(os.path.join(out, "a2s_cat2.json"), "w"))
    print("a2s_cat2: %d bytes, %d bones" % (nbytes, len(mix)), flush=True)

    picture(Qa, Qs, moved, chains, tgt, os.path.join(out, "home2_stand.png"))


def picture(Qa, Qs, moved, chains, tgt, path):
    """before and after, front / side / below; the bent legs' skeletons and the targets"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    for r, (Y, title) in enumerate(((Qa, "now (own frame 50)"), (Qs, "squared"))):
        for c, (a, b, view) in enumerate(((0, 1, "front (x, y)"), (2, 1, "side (z, y)"), (0, 2, "from below (x, z)"))):
            ax = axes[r, c]
            m = np.ones(len(Y), bool) if b == 1 else Y[:, 1] > Y[:, 1].max() - 0.12
            ax.scatter(Y[m, a][::2], Y[m, b][::2], s=0.3, c=np.where(moved[m], 0.0, 1.0)[::2], cmap="coolwarm", alpha=0.4)
            if r == 0:
                for ch in chains:
                    ax.plot(ch[:, a], ch[:, b], "k-o", ms=3, lw=1.5)
            for k in range(4):
                ax.plot(tgt[k, a], tgt[k, b], "g*", ms=12)
                ax.annotate(PAWS[k][4:], (tgt[k, a], tgt[k, b]), fontsize=9)
            ax.set_aspect("equal")
            ax.set_title("%s, %s" % (title, view))
            if b == 1:
                ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(path, dpi=65)

if __name__ == "__main__":
    main()
