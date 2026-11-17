"""Feet and pose check (work/pets/PROTOCOL_CAT.md): square the grey and white cat's rest pose, then redo its retarget.

Its rest pose (its own video's frame 50, lined up with Pikachu's rest) has the +x front paw 8 cm up and the hind paws
one behind the other, so on Pikachu's motion that paw never comes down and the front legs follow a mix of Pikachu's two.
No frame of its own video stands square, so the rest mesh is bent as rigidly as possible (ARAP, cotangent weights):
each paw moves to a square stance (Pikachu's rest paw places about its centre, scaled by the length ratio, at the height
of the grey cat's lowest paw), the body further than 18 cm from every paw stays, the legs in between bend. Then the
correspondence bend and the bone mixes are redone as in home2_retarget.py (MODE=region).
CPU only (numpy, scipy, matplotlib). Run with OPENBLAS_NUM_THREADS=1 on the login node.

usage: python home2_square.py PAGE_DIR UV_NPZ SKEL_NPZ OUT_DIR
  PAGE_DIR: the page's a2s_cat, a2s_cat2 (the current grey rest, already in Pikachu's frame), a2s_walks (.bin/.json)
  UV_NPZ, SKEL_NPZ: the grey cat's canonical mesh and its own motion (for the stretch read-out, as in home2_retarget)
Writes OUT_DIR/a2s_cat2.bin + .json (the squared rest, same faces, UVs and skinning, new mixes), home2_square.json,
home2_square.png.
"""
import json
import os
import sys

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.sparse.linalg import factorized
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feet_check import LMS, PAWS, landmark_sets  # noqa: E402
from home2_retarget import nonrigid, region_mix, region_retarget  # noqa: E402
from home2_transfer import edges_of, lbs, pack, read_blobs, stretch  # noqa: E402

HANDLE_R = float(os.environ.get("HANDLE_R", 0.02))  # each paw: its tip vertices and those this close move with the tip
BODY_R = float(os.environ.get("BODY_R", 0.18))  # vertices further than this from every paw tip stay put
ITERS = 20


def cot_weights(X, faces):
    """edge list and cotangent weights (half the sum of the two opposite angles' cotangents; kept positive)"""
    w = {}
    for a, b, c in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
        u = X[faces[:, a]] - X[faces[:, c]]
        v = X[faces[:, b]] - X[faces[:, c]]
        cot = (u * v).sum(1) / np.maximum(np.linalg.norm(np.cross(u, v), axis=1), 1e-12)
        i, j = np.minimum(faces[:, a], faces[:, b]), np.maximum(faces[:, a], faces[:, b])
        for key, val in zip(zip(i.tolist(), j.tolist()), (0.5 * cot).tolist()):
            w[key] = w.get(key, 0.0) + val
    e = np.array(list(w.keys()), dtype=np.int64)
    ww = np.array(list(w.values()))
    return e, np.maximum(ww, 0.05 * np.median(np.abs(ww)))


def geodesic_from(X, e, src):
    L = np.linalg.norm(X[e[:, 0]] - X[e[:, 1]], axis=1)
    G = coo_matrix((np.r_[L, L], (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])), shape=(len(X), len(X))).tocsr()
    return dijkstra(G, indices=src, min_only=True)


def arap(X, faces, known, pos):
    """as-rigid-as-possible: vertices in `known` (bool) go to pos[known]; the rest follow"""
    e, w = cot_weights(X, faces)
    n = len(X)
    Lw = coo_matrix((np.r_[w, w], (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])), shape=(n, n)).tocsr()
    L = (coo_matrix((np.asarray(Lw.sum(1)).ravel(), (np.arange(n), np.arange(n))), shape=(n, n)) - Lw).tocsr()
    U, K = np.where(~known)[0], np.where(known)[0]
    solve = factorized(L[U][:, U].tocsc())
    LUK = L[U][:, K]
    d0 = X[e[:, 0]] - X[e[:, 1]]
    Y = X.copy()
    Y[K] = pos[K]
    R = np.broadcast_to(np.eye(3), (n, 3, 3)).copy()
    for it in range(ITERS + 1):
        # global step: L Y = b, b from the current rotations
        Re = 0.5 * (R[e[:, 0]] + R[e[:, 1]])
        t = w[:, None] * np.einsum("eij,ej->ei", Re, d0)
        b = np.zeros((n, 3))
        np.add.at(b, e[:, 0], t)
        np.add.at(b, e[:, 1], -t)
        rhs = b[U] - LUK @ Y[K]
        Y[U] = np.stack([solve(rhs[:, k]) for k in range(3)], 1)
        if it == ITERS:
            break
        # local step: each vertex's best rotation
        d1 = Y[e[:, 0]] - Y[e[:, 1]]
        S = np.zeros((n, 3, 3))
        o = w[:, None, None] * np.einsum("ei,ej->eij", d1, d0)
        np.add.at(S, e[:, 0], o)
        np.add.at(S, e[:, 1], o)
        Us, _, Vt = np.linalg.svd(S)
        R = Us @ Vt
        bad = np.linalg.det(R) < 0
        Us[bad, :, 2] *= -1
        R[bad] = Us[bad] @ Vt[bad]
    return Y


def main():
    page_dir, uv_npz, skel_npz, out = sys.argv[1:5]
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

    # ---- targets: a square stance
    lp, lg = landmark_sets(P, fp), landmark_sets(Qa, fg)
    tipP = np.stack([P[s].mean(0) for s in lp[:4]])
    tipG = np.stack([Qa[s].mean(0) for s in lg[:4]])
    lenP = np.linalg.norm(P[lp[4]].mean(0) - tipP[2:].mean(0))
    lenG = np.linalg.norm(Qa[lg[4]].mean(0) - tipG[2:].mean(0))
    k = lenG / lenP
    # each paw straight below where its own leg joins the body (the leg 10 to 14 cm up from the paw tip, along the
    # surface); each pair side by side (same front-to-back place, at least 5 cm apart sideways, each paw on its own
    # leg's side); all four at the height of the grey cat's lowest paw (y down)
    e = edges_of(fg)
    geo = np.stack([geodesic_from(Qa, e, s) for s in lg[:4]])  # (4, V)
    top = np.stack([Qa[(geo[p] > 0.10) & (geo[p] < 0.14)].mean(0) for p in range(4)])
    tgt = np.zeros((4, 3))
    for a, b in ((0, 1), (2, 3)):
        mid_x, half = top[[a, b], 0].mean(), max(abs(top[a, 0] - top[b, 0]) / 2, 0.025)
        for p in (a, b):
            tgt[p, 0] = mid_x + (half if top[p, 0] > mid_x else -half)
            tgt[p, 2] = top[[a, b], 2].mean()
    tgt[:, 1] = tipG[:, 1].max()
    move = tgt - tipG
    print("length ratio %.3f; leg tops: " % k + ", ".join("%s %s" % (n, np.round(t, 3)) for n, t in zip(PAWS, top)), flush=True)
    print("paw moves (cm): " + ", ".join("%s %s" % (n, np.round(100 * m, 1)) for n, m in zip(PAWS, move)), flush=True)

    # ---- held and free vertices
    known = np.zeros(len(Qa), bool)
    pos = Qa.copy()
    for p in range(4):
        h = geo[p] < HANDLE_R
        assert not (known & h).any(), "paw handles overlap"
        known |= h
        pos[h] = Qa[h] + move[p]
    body = geo.min(0) > BODY_R
    known |= body
    print("held: %d paw vertices, %d body vertices; %d free of %d" % (
        (geo.min(0) < HANDLE_R).sum(), body.sum(), (~known).sum(), len(Qa)), flush=True)
    Qs = arap(Qa, fg, known, pos)

    # ---- checks on the squaring itself
    leg = geo.min(0) <= BODY_R
    le = e[leg[e[:, 0]] & leg[e[:, 1]]]
    st = stretch(Qa, Qs, le)
    sq_share = float((st > np.log(1.3)).mean())
    picture(P, Qa, Qs, fg, tipG, tgt, os.path.join(out, "home2_square.png"))
    print("squaring: %.2f%% of leg edges stretched over 30%%" % (100 * sq_share), flush=True)
    lg2 = landmark_sets(Qs, fg)
    tip2 = np.stack([Qs[s].mean(0) for s in lg2[:4]])
    same_paws = all(len(np.intersect1d(a, b)) >= 10 for a, b in zip(lg[:4], lg2[:4]))
    tip_off = np.linalg.norm(tip2 - tgt, axis=1)
    at_targets = bool((tip_off <= 0.02).all())
    print("squaring: %.2f%% of leg edges stretched over 30%% (p99 %.3f log); new paw tips " % (
        100 * sq_share, np.percentile(st, 99)) + ", ".join("%s %s" % (n, np.round(t, 3)) for n, t in zip(PAWS, tip2)) +
        "; same paws found: %s; off their targets (cm) %s" % (same_paws, np.round(100 * tip_off, 1)), flush=True)
    if not (sq_share <= 0.02 and at_targets):
        print("NOT KEPT: squaring fails its own check", flush=True)
    spread = tip2[:, 1].max() - tip2[:, 1].min()
    apart = min(abs(tip2[0, 0] - tip2[1, 0]), abs(tip2[2, 0] - tip2[3, 0]))
    stagger = max(abs(tip2[0, 2] - tip2[1, 2]), abs(tip2[2, 2] - tip2[3, 2]))
    print("square: level within %.1f cm, pairs %.1f cm apart sideways, %.1f cm front to back" % (
        100 * spread, 100 * apart, 100 * stagger), flush=True)

    # ---- the retarget again, from the squared rest (as home2_retarget.py, MODE=region)
    tree_p = cKDTree(P)
    X = nonrigid(Qs, fg, P, fp, tree_p, [30.0, 10.0, 3.0])
    mix, C, share = region_mix(Qs, X, ig, wg, P, ip, wp, B)
    sk, z = np.load(skel_npz), np.load(uv_npz)
    Qc = z["pos"].astype(np.float64)
    own = sk["own"].astype(np.float64)
    e1 = edges_of(fp)
    fr = np.linspace(0, len(rec) - 1, 40).round().astype(int)
    fw = np.linspace(0, len(walks) - 1, 40).round().astype(int)
    st_g = [stretch(Qs, lbs(Qs, ig, wg, T), e) for T in region_retarget(np.concatenate([rec[fr], walks[fw]]), mix, C)]
    st_own = [stretch(Qc, lbs(Qc, ig, wg, own[f]), e) for f in np.linspace(0, len(own) - 1, 40).round().astype(int)]
    st_g, st_own = np.concatenate(st_g), np.concatenate(st_own)
    ratio = float(np.percentile(st_g, 99) / np.percentile(st_own, 99))
    check = {"length_ratio": float(k), "paw_moves_cm": dict(zip(PAWS, (100 * move).round(2).tolist())),
             "held_paw_vertices": int((geo.min(0) < HANDLE_R).sum()), "held_body_vertices": int(body.sum()),
             "squaring_leg_edges_over_30pct": sq_share, "squaring_ok": sq_share <= 0.02 and at_targets, "same_paws_found": bool(same_paws),
             "paws_off_targets_cm": (100 * tip_off).round(2).tolist(), "body_r": BODY_R, "handle_r": HANDLE_R,
             "square_level_cm": 100 * float(spread), "square_pairs_apart_cm": 100 * float(apart),
             "square_front_to_back_cm": 100 * float(stagger),
             "stretch_p99_retargeted_to_own": ratio, "readout": "moves cleanly" if ratio <= 1.5 else (
                 "some stretching" if ratio <= 3 else "tears"),
             "share_edges_over_30pct_retargeted": float((st_g > np.log(1.3)).mean()),
             "share_edges_over_30pct_own": float((st_own > np.log(1.3)).mean())}
    json.dump(check, open(os.path.join(out, "home2_square.json"), "w"), indent=1)
    print(json.dumps(check, indent=1), flush=True)
    meta, nbytes = pack(os.path.join(out, "a2s_cat2.bin"), [
        ("faces", fg.astype(np.uint32)), ("pos", Qs.astype(np.float32)), ("uv", uv),
        ("skin_idx", ig.astype(np.uint8)), ("skin_w", wg.astype(np.float32))])
    meta_json = {k_: v for k_, v in gm.items() if k_ not in ("blobs", "bytes", "chunks", "mix", "centre")}
    meta_json.update({"blobs": meta, "bytes": nbytes, "mix": mix, "centre": np.round(C, 6).tolist(),
                      "squared": "rest pose bent as rigidly as possible to stand square (home2_square.py)"})
    json.dump(meta_json, open(os.path.join(out, "a2s_cat2.json"), "w"))
    print("a2s_cat2: %d bytes" % nbytes, flush=True)


def picture(P, Qa, Qs, fg, tipG, tgt, path):
    """rest before and after (side, front, top) over Pikachu's rest; black x: paw tips before; red o: targets"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for r, (Y, title) in enumerate(((Qa, "grey rest before"), (Qs, "grey rest squared"))):
        for c, (a, b) in enumerate(((2, 1), (0, 1), (0, 2))):
            ax = axes[r, c]
            ax.scatter(P[::3, a], P[::3, b], s=0.3, c="tab:orange", alpha=0.25)
            ax.scatter(Y[::3, a], Y[::3, b], s=0.3, c="tab:blue", alpha=0.3)
            ax.scatter(tipG[:, a], tipG[:, b], s=30, c="k", marker="x")
            ax.scatter(tgt[:, a], tgt[:, b], s=30, facecolors="none", edgecolors="r")
            ax.set_aspect("equal")
            if b == 1:
                ax.invert_yaxis()
            ax.set_title("%s (%s)" % (title, ("side", "front", "top")[c]))
    fig.tight_layout()
    fig.savefig(path, dpi=70)


if __name__ == "__main__":
    main()
