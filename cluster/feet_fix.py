"""Paw fix (work/pets/PROTOCOL_CAT.md, "Feet and pose check"): hold each paw in place while it is down, for both cats.

Contacts are Pikachu's (a paw is down when its tip is at most 2 cm above Pikachu's lowest point), so both cats step
together. While a paw is down it is held at the median of its places over that stretch of contact (sideways) and no
higher than its rest offset above the cat's lowest point (height); the correction eases in and out over 2 frames and is
capped at 4 cm. It moves the paw and lower leg: weight 1 at the paw tip, falling linearly (along the surface) to 0 at
60% of the leg's height in the rest pose. The page adds it after posing (index.html). Also checks the page's floor
against the room scan under the cat's paws. CPU only.

usage: python feet_fix.py PAGE_DIR CHECK_DIR UV_NPZ SKEL_NPZ OUT_DIR [FLOOR_DIR]
  CHECK_DIR: feet_check.py's output (feet_frames.npz); UV_NPZ, SKEL_NPZ: the grey cat's own video (stretch read-out);
  FLOOR_DIR: feet_floor.py's output, packed in too when given. STEP=0.03 (default): held only while moving less than
  3 cm sideways in 0.1 s; STEP=0: the first form, every down stretch held
Writes OUT_DIR/feet_fix.bin + .json (per recorded and walk frame and paw: each cat's correction in the cat's own frame;
each cat's leg weights), feet_fix_check.json and feet_fix.png.
"""
import json
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feet_check import DOWN, FLOAT, PAWS, SKID, Page, skin  # noqa: E402
from home2_retarget import region_retarget  # noqa: E402
from home2_square import geodesic_from  # noqa: E402
from home2_transfer import edges_of, lbs, pack, stretch  # noqa: E402

CAP = 0.04
EASE = (2 / 3, 1 / 3)
LEG_FRAC = 0.6
STEP = float(os.environ.get("STEP", 0.03))  # 0: the first form (every down stretch held)


def runs_of(mask):
    """start, end (inclusive) of each run of True"""
    d = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0] - 1))


def corrections(paw, low, down, track, h0):
    """paw (N,4,3) world places, low (N,) lowest point (y down), down (N,4) contacts, track (N,) -> offsets (N,4,3)"""
    N = len(paw)
    off = np.zeros((N, 4, 3))
    starts = np.r_[0, np.where(np.diff(track) != 0)[0] + 1, N]
    for a, b in zip(starts[:-1], starts[1:]):
        for p in range(4):
            rr = runs_of(down[a:b, p])
            held = np.zeros(b - a, bool)
            for s, e in rr:
                held[s:e + 1] = True
            for s, e in rr:
                seg = paw[a + s:a + e + 1, p]
                tgt_xz = np.median(seg[:, [0, 2]], 0)
                o = np.zeros((e - s + 1, 3))
                o[:, 0], o[:, 2] = tgt_xz[0] - seg[:, 0], tgt_xz[1] - seg[:, 2]
                o[:, 1] = np.maximum(0.0, (low[a + s:a + e + 1] - h0[p]) - seg[:, 1])  # lower only (y down)
                n = np.linalg.norm(o, axis=1, keepdims=True)
                o *= np.minimum(1.0, CAP / np.maximum(n, 1e-9))
                off[a + s:a + e + 1, p] = o
                for k, wgt in enumerate(EASE):  # ease out after and in before, only on frames not held themselves
                    if e + 1 + k < b - a and not held[e + 1 + k]:
                        off[a + e + 1 + k, p] += wgt * o[-1]
                    if s - 1 - k >= 0 and not held[s - 1 - k]:
                        off[a + s - 1 - k, p] += wgt * o[0]
    return off


def leg_weights(X, faces, sets):
    """per paw: vertices and weights, 1 at the paw tip falling linearly along the surface to 0 at 60% of the leg's
    height in the rest pose (paw tip to the body's centre, y down)"""
    e = edges_of(faces)
    c = X.mean(0)
    out = []
    for s in sets[:4]:
        g = geodesic_from(X, e, s)
        r = LEG_FRAC * (X[s, 1].mean() - c[1])
        w = np.clip(1 - g / r, 0, 1)
        v = np.where(w > 0)[0]
        out.append((v, w[v], r))
    return out


def slide_share(paw, h, track, kind, k):
    """M1's share of down steps skidding over 1.5 cm, the number of steps, and the share skidding 1.5 to 3 cm (slow
    skating, what the hold is for)"""
    same = np.r_[track[1:] == track[:-1], False]
    d = h <= DOWN
    step = d[:-1] & d[1:] & same[:-1, None] & (kind[:-1] == k)[:, None]
    skid = np.linalg.norm((paw[1:] - paw[:-1])[..., [0, 2]], axis=-1)[step]
    return float((skid > SKID).mean()), int(step.sum()), float(((skid > SKID) & (skid <= 0.03)).mean())


def main():
    page_dir, check_dir, uv_npz, skel_npz, out = sys.argv[1:6]
    floor_dir = sys.argv[6] if len(sys.argv) > 6 else None
    os.makedirs(out, exist_ok=True)
    page = Page(page_dir)
    F = np.load(os.path.join(check_dir, "feet_frames.npz"))
    track, kind, low = F["track"], F["kind"], F["lowW"]
    H = np.concatenate([tr["H"] for tr in page.tracks])
    R = np.concatenate([tr["root"] for tr in page.tracks])
    assert len(H) == len(track) and (track[:page.tracks[0]["bones"].shape[0]] == 0).all()
    nrec = int((kind == "rec").sum())
    assert (kind[:nrec] == "rec").all() and (kind[nrec:] == "walk").all()
    downP = F["hP"] <= DOWN
    # second form (the first held every down stretch and hit the cap): held only while down and moving less than
    # STEP sideways in 0.1 s on Pikachu (a faster move is a step and ends the hold)
    same = np.r_[track[1:] == track[:-1], False]
    stepP = np.r_[np.linalg.norm((F["pawP"][1:] - F["pawP"][:-1])[..., [0, 2]], axis=-1), np.zeros((1, 4))]
    hold = downP & (stepP < STEP) & same[:, None] if STEP > 0 else downP
    print("held: %.1f%% of Pikachu's down paw-frames (step limit %.0f cm)" % (100 * hold[downP].mean(), 100 * STEP), flush=True)
    h0 = {c: np.array([X[s, 1].max() - X[s, 1].mean() for s in sets[:4]])
          for c, X, sets in (("p", page.P, page.lp), ("g", page.Q, page.lg))}
    print("paw tip offsets above its lowest vertex (rest, cm): Pikachu %s, grey %s" % (
        np.round(100 * h0["p"], 2), np.round(100 * h0["g"], 2)), flush=True)
    offP = corrections(F["pawP"], low, hold, track, h0["p"])
    offG = corrections(F["pawG"], low, hold, track, h0["g"])
    pawP2, pawG2 = F["pawP"] + offP, F["pawG"] + offG
    hP2, hG2 = low[:, None] - pawP2[..., 1], low[:, None] - pawG2[..., 1]

    # ---- the measures again, after the fix
    res = {"cap_m": CAP, "leg_frac": LEG_FRAC}
    for cat, paw, h, paw2, h2, off in (("pikachu", F["pawP"], F["hP"], pawP2, hP2, offP),
                                       ("grey", F["pawG"], F["hG"], pawG2, hG2, offG)):
        n = np.linalg.norm(off, axis=-1)
        for k in ("rec", "walk"):
            b, nb, sb = slide_share(paw, h, track, kind, k)
            a, na, sa = slide_share(paw2, h2, track, kind, k)
            m = (kind == k)[:, None] & downP
            res["%s %s" % (cat, k)] = {"slide_share_before": b, "slide_share_after": a, "steps_before": nb, "steps_after": na,
                                       "skating_1.5_to_3cm_before": sb, "skating_1.5_to_3cm_after": sa,
                                       "skating_readout_after": "planted" if sa < 0.05 else "still skates",
                                       "correction_cm_median_while_down": 100 * float(np.median(n[m])),
                                       "correction_cm_p90_while_down": 100 * float(np.percentile(n[m], 90)),
                                       "share_capped_while_down": float((n[m] >= CAP - 1e-6).mean()),
                                       "slide_readout_after": "planted" if a < 0.05 else "still slides"}
    for k in ("rec", "walk"):
        m = (kind == k)[:, None] & (hP2 <= DOWN)
        res["grey floating %s" % k] = {"share_over_3cm_before": float((F["hG"][(kind == k)[:, None] & downP] > FLOAT).mean()),
                                       "share_over_3cm_after": float((hG2[m] > FLOAT).mean()),
                                       "per_paw_after": [float((hG2[:, p][m[:, p]] > FLOAT).mean()) for p in range(4)]}

    # ---- the floor: the page's (2nd percentile of the scan) against the scan right under each paw that is down
    import json as _j
    from home2_transfer import read_blobs
    rm = _j.load(open(os.path.join(page_dir, "room.json")))
    room = read_blobs(os.path.join(page_dir, "room.bin"), rm)["pos"].astype(np.float64)
    near = room[np.abs(room[:, 1] - page.floor) < 0.15]
    tree = cKDTree(near[:, [0, 2]])
    m = downP & (np.abs(page.floor - low) <= 0.10)[:, None]
    pts = F["pawP"][m]
    sel = pts[np.random.default_rng(0).choice(len(pts), min(20000, len(pts)), replace=False)]
    local = np.array([np.median(near[i, 1]) if len(i) >= 5 else np.nan for i in tree.query_ball_point(sel[:, [0, 2]], 0.06)])
    ok = np.isfinite(local)
    res["floor"] = {"page_floor_m": page.floor, "paws_checked": int(ok.sum()),
                    "scan_under_paw_minus_page_floor_cm_median": 100 * float(np.median(local[ok] - page.floor)),
                    "scan_under_paw_minus_page_floor_cm_p10_p90": [100 * float(np.percentile(local[ok] - page.floor, q)) for q in (10, 90)],
                    "paw_above_scan_under_it_cm_median": 100 * float(np.median(local[ok] - sel[ok, 1])),
                    "paw_above_scan_under_it_cm_p10_p90": [100 * float(np.percentile(local[ok] - sel[ok, 1], q)) for q in (10, 90)],
                    "paw_above_page_floor_cm_median": 100 * float(np.median(page.floor - sel[ok, 1]))}

    # ---- leg weights, and stretch with the fix on sampled frames
    legP = leg_weights(page.P, page.fp, page.lp)
    legG = leg_weights(page.Q, page.fg, page.lg)
    print("leg weight radius (cm): Pikachu %s, grey %s" % ([round(100 * r, 1) for _, _, r in legP],
                                                           [round(100 * r, 1) for _, _, r in legG]), flush=True)
    sk, z = np.load(skel_npz), np.load(uv_npz)
    Qc, own = z["pos"].astype(np.float64), sk["own"].astype(np.float64)
    eP, eG = edges_of(page.fp), edges_of(page.fg)
    rng = np.random.default_rng(1)
    rows = np.r_[rng.choice(np.where(kind == "rec")[0], 40, replace=False), rng.choice(np.where(kind == "walk")[0], 40, replace=False)]
    rows = rows[np.argsort(rows)]
    stP0, stP1, stG = [], [], []
    for i in rows:
        tr = page.tracks[track[i]]
        t = int(F["t"][i])
        T = tr["bones"][t:t + 1].astype(np.float64)
        Hi, down = tr["H"][t], tr["H"][t][1]
        XP = lbs(page.P, page.ip, page.wp, T[0])
        XG = lbs(page.Q, page.ig, page.wg, region_retarget(T, page.mix, page.C)[0])
        XG += ((XP[::3] @ down).max() - (XG[::3] @ down).max()) * down
        XP1, XG1 = XP.copy(), XG.copy()
        for p in range(4):
            for X1, (v, w, _), off in ((XP1, legP[p], offP), (XG1, legG[p], offG)):
                X1[v] += w[:, None] * (Hi.T @ off[i, p])[None]
        stP0.append(stretch(page.P, XP, eP))
        stP1.append(stretch(page.P, XP1, eP))
        stG.append(stretch(page.Q, XG1, eG))
    st_own = np.concatenate([stretch(Qc, lbs(Qc, page.ig, page.wg, own[f]), eG) for f in np.linspace(0, len(own) - 1, 40).round().astype(int)])
    stP0, stP1, stG = np.concatenate(stP0), np.concatenate(stP1), np.concatenate(stG)
    ratio = float(np.percentile(stG, 99) / np.percentile(st_own, 99))
    res["stretch"] = {"grey_p99_with_fix_over_own_video": ratio, "grey_readout": "moves cleanly" if ratio <= 1.5 else (
        "some stretching" if ratio <= 3 else "tears"),
        "pikachu_p99_before": float(np.percentile(stP0, 99)), "pikachu_p99_with_fix": float(np.percentile(stP1, 99)),
        "pikachu_edges_over_30pct_before": float((stP0 > np.log(1.3)).mean()),
        "pikachu_edges_over_30pct_with_fix": float((stP1 > np.log(1.3)).mean())}
    json.dump(res, open(os.path.join(out, "feet_fix_check.json"), "w"), indent=1)
    print(json.dumps(res, indent=1), flush=True)

    # ---- for the page: corrections in each cat's own frame (world = H local + root), and the leg weights
    loc = lambda off: np.einsum("nji,npj->npi", H, off)  # noqa: E731  H^T per frame
    oP, oG = loc(offP).astype(np.float16), loc(offG).astype(np.float16)
    blobs = [("rec_p", oP[:nrec]), ("walk_p", oP[nrec:]), ("rec_g", oG[:nrec]), ("walk_g", oG[nrec:])]
    if floor_dir:  # the floor fix: raise or lower the cat (room's y down), from feet_floor.py
        fl = np.load(os.path.join(floor_dir, "feet_floor.npz"))
        blobs += [("rec_dy", fl["rec_dy"].astype(np.float16)), ("walk_dy", fl["walk_dy"].astype(np.float16))]
    counts = {}
    for c, leg in (("p", legP), ("g", legG)):
        blobs.append(("leg_%s_idx" % c, np.concatenate([v for v, _, _ in leg]).astype(np.uint16)))
        blobs.append(("leg_%s_w" % c, np.concatenate([w for _, w, _ in leg]).astype(np.float16)))
        counts[c] = [int(len(v)) for v, _, _ in leg]
    meta, nbytes = pack(os.path.join(out, "feet_fix.bin"), blobs)
    json.dump({"blobs": meta, "bytes": nbytes, "paws": PAWS, "leg_counts": counts, "rec_frames": nrec,
               "walk_frames": int(len(kind) - nrec), "units": "metres, the cat's own frame (y down)",
               "from": "paws held while down (Pikachu's contacts), capped at 4 cm; feet_fix.py"},
              open(os.path.join(out, "feet_fix.json"), "w"))
    print("feet_fix: %d bytes" % nbytes, flush=True)

    # ---- picture: walk 3, 0 to 20 s: paw heights and sideways drift, before and after
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    w3 = [i for i, tr in enumerate(page.tracks) if tr["name"] == "walk 3"][0]
    m = (track == w3) & (F["t"] < 200)
    tt = F["t"][m] / 10
    fig, axes = plt.subplots(4, 2, figsize=(16, 11), sharex=True)
    for p in range(4):
        for c, (nm, paw, paw2, col) in enumerate((("Pikachu", F["pawP"], pawP2, "tab:orange"), ("grey cat", F["pawG"], pawG2, "tab:blue"))):
            ax = axes[p, c]
            for X, ls, lab in ((paw, ":", "before"), (paw2, "-", "after")):
                hh = low[m] - X[m, p, 1]
                sk_ = np.r_[0, np.linalg.norm(np.diff(X[m, p][:, [0, 2]], axis=0), axis=1)]
                ax.plot(tt, 100 * hh, ls, c=col, label="height, " + lab)
                ax.plot(tt, 100 * sk_, ls, c="k", lw=0.7, label="sideways move per 0.1 s, " + lab)
            ax.set_ylim(-1, 15)
            ax.set_ylabel("%s\n%s (cm)" % (nm, PAWS[p]))
    axes[0, 0].legend(fontsize=7)
    axes[-1, 0].set_xlabel("walk 3, seconds")
    axes[-1, 1].set_xlabel("walk 3, seconds")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "feet_fix.png"), dpi=70)


if __name__ == "__main__":
    main()
