"""Paws that stay planted (work/pets/PROTOCOL_CAT.md, "Paws that stay planted"): short lifted steps where the motion glides.

The earlier holds (feet_fix.py) failed because the body glides while the legs stay still, so a paw held where it came
down soon needs more than the 4 cm the leg can give. Here, per track and per cat, the four paws go through one state
machine, one frame (0.1 s) at a time:
- planted: a paw that comes down (Pikachu's contacts: tip at most 2 cm above Pikachu's lowest point) is held where it
  came down (sideways), and no higher than its rest offset above the cat's lowest point (lowering capped at 2 cm);
- step: when holding it would need more than 4 cm sideways, the paw spends two frames in the air (3 cm up, a third and
  two thirds of the way) and lands at its natural place plus a lead (its natural move over the next 3 frames, capped
  at 2.4 cm);
- gait rule: at most two paws in the air at once, never both fronts or both hinds; the paw needing the largest
  correction goes first; a paw that must wait is held at the 4 cm limit (it slides with the body);
- the motion's own lifts (natural tip over 2 cm up): the paw follows the motion again, easing out over 2 frames.
The correction moves the paw and lower leg (feet_fix.leg_weights). CPU only.

usage: python feet_step.py PAGE_DIR CHECK_DIR UV_NPZ SKEL_NPZ OUT_DIR
  CHECK_DIR: feet_check.py's output (feet_frames.npz); UV_NPZ, SKEL_NPZ: the grey cat's own video (stretch read-out)
Writes OUT_DIR/feet_step.bin + .json (per recorded and walk frame and paw: each cat's correction in the cat's own
frame; each cat's leg weights), feet_step_check.json and feet_step.png.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feet_check import DOWN, FLOAT, PAWS, Page  # noqa: E402
from feet_fix import leg_weights, slide_share  # noqa: E402
from home2_retarget import region_retarget  # noqa: E402
from home2_transfer import edges_of, lbs, pack, stretch  # noqa: E402

CAP = 0.04       # sideways correction a leg can give before the paw steps
LOWER = 0.02     # largest lowering of a planted paw
LIFT = 0.03      # paw height in the air during a step
LEAD_N, LEAD_CAP = 3, 0.024
EASE = (2 / 3, 1 / 3)
GIRDLE = (1, 0, 3, 2)  # the other paw of the same pair (fronts 0-1, hinds 2-3)
FREE, PLANTED, AIR = 0, 1, 2


def steps(paw, low, down, h0):
    """paw (n,4,3) natural world places (y down), low (n,) lowest point, down (n,4) contacts, h0 (4,) paw tip offset
    above its lowest vertex at rest -> offsets (n,4,3), stats"""
    n = len(paw)
    off = np.zeros((n, 4, 3))
    xz = paw[..., [0, 2]]
    state = np.full(4, FREE)
    anchor = np.zeros((4, 2))
    air_from, air_to, air_t0 = np.zeros((4, 2)), np.zeros((4, 2)), np.zeros(4, int)
    ease_left, ease_from = np.zeros(4, int), np.zeros((4, 3))
    planted_len = np.zeros(4, int)
    st = {"planted_runs": [], "run_ends": [], "steps": 0, "waits": 0, "planted_frames": 0}

    def lowered(t, p):  # height offset of a planted paw (y down: positive lowers it)
        return float(np.clip((low[t] - h0[p]) - paw[t, p, 1], 0.0, LOWER))

    def end_plant(p, why):
        if planted_len[p]:
            st["planted_runs"].append(int(planted_len[p]))
            st["run_ends"].append(why)
        planted_len[p] = 0

    for t in range(n):
        in_air = [p for p in range(4) if state[p] == AIR and t - air_t0[p] < 2]
        # swings that land on this frame
        for p in range(4):
            if state[p] == AIR and t - air_t0[p] >= 2:
                if down[t, p]:
                    state[p], anchor[p] = PLANTED, air_to[p]
                else:
                    state[p], ease_left[p] = FREE, len(EASE)
                    ease_from[p] = np.r_[air_to[p][0] - xz[t, p, 0], 0.0, air_to[p][1] - xz[t, p, 1]][[0, 1, 2]]
        want = []
        for p in range(4):
            if state[p] == FREE:
                o = np.zeros(3)
                if ease_left[p] > 0:
                    o = EASE[len(EASE) - ease_left[p]] * ease_from[p]
                    ease_left[p] -= 1
                if down[t, p]:  # comes down where it is shown now: no jump
                    state[p], anchor[p] = PLANTED, xz[t, p] + o[[0, 2]]
                else:
                    off[t, p] = o
            if state[p] == PLANTED:
                if not down[t, p]:  # the motion lifts it: follow the motion again
                    end_plant(p, "lift")
                    state[p], ease_left[p] = FREE, len(EASE) - 1
                    ease_from[p] = off[t - 1, p] if t else 0
                    off[t, p] = EASE[0] * ease_from[p]
                    continue
                d = anchor[p] - xz[t, p]
                if np.linalg.norm(d) > CAP:
                    want.append((float(np.linalg.norm(d)), p))
                else:
                    off[t, p] = (d[0], lowered(t, p), d[1])
                    planted_len[p] += 1
                    st["planted_frames"] += 1
        for _, p in sorted(want, reverse=True):
            if len(in_air) < 2 and GIRDLE[p] not in in_air:
                end_plant(p, "step")
                prev = off[t - 1, p] if t else np.zeros(3)
                start = xz[t - 1, p] + prev[[0, 2]] if t else xz[t, p]  # where it is shown on the frame before
                tl = min(t + 2, n - 1)
                lead = xz[min(tl + LEAD_N, n - 1), p] - xz[tl, p]
                ln = np.linalg.norm(lead)
                if ln > LEAD_CAP:
                    lead *= LEAD_CAP / ln
                state[p], air_t0[p], air_from[p], air_to[p] = AIR, t, start, xz[tl, p] + lead
                in_air.append(p)
                st["steps"] += 1
            else:  # waits at the limit and slides with the body
                d = anchor[p] - xz[t, p]
                d *= CAP / np.linalg.norm(d)
                anchor[p] = xz[t, p] + d
                off[t, p] = (d[0], lowered(t, p), d[1])
                planted_len[p] += 1
                st["planted_frames"] += 1
                st["waits"] += 1
        for p in range(4):
            if state[p] == AIR and t - air_t0[p] < 2:
                f = (t - air_t0[p] + 1) / 3
                place = air_from[p] + f * (air_to[p] - air_from[p])
                up = min(0.0, (low[t] - h0[p] - LIFT) - paw[t, p, 1])  # y down: negative lifts it
                off[t, p] = (place[0] - xz[t, p, 0], up, place[1] - xz[t, p, 1])
    for p in range(4):
        end_plant(p, "end")
    return off, st


def natural_runs(down, track, kind, k):
    """lengths (frames) of the motion's own down stretches (Pikachu's contacts), per paw, on tracks of kind k"""
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
    H = np.concatenate([tr["H"] for tr in page.tracks])
    assert len(H) == len(track)
    nrec = int((kind == "rec").sum())
    assert (kind[:nrec] == "rec").all() and (kind[nrec:] == "walk").all()
    downP = F["hP"] <= DOWN
    h0 = {c: np.array([X[s, 1].max() - X[s, 1].mean() for s in sets[:4]])
          for c, X, sets in (("p", page.P, page.lp), ("g", page.Q, page.lg))}
    starts = np.r_[0, np.where(np.diff(track) != 0)[0] + 1, len(track)]
    offs, stats = {}, {}
    for c, paw in (("p", F["pawP"]), ("g", F["pawG"])):
        o = np.zeros_like(paw)
        st = {k: {"planted_runs": [], "run_ends": [], "steps": 0, "waits": 0, "planted_frames": 0} for k in ("rec", "walk")}
        for a, b in zip(starts[:-1], starts[1:]):
            oo, s = steps(paw[a:b], low[a:b], downP[a:b], h0[c])
            o[a:b] = oo
            k = kind[a]
            for key in ("steps", "waits", "planted_frames"):
                st[k][key] += s[key]
            st[k]["planted_runs"] += s["planted_runs"]
            st[k]["run_ends"] += s["run_ends"]
        offs[c], stats[c] = o, st
        print("cat %s done" % c, flush=True)
    pawP2, pawG2 = F["pawP"] + offs["p"], F["pawG"] + offs["g"]
    hP2, hG2 = low[:, None] - pawP2[..., 1], low[:, None] - pawG2[..., 1]

    # ---- measures after the steps
    res = {"cap_m": CAP, "lower_cap_m": LOWER, "lift_m": LIFT, "lead_frames": LEAD_N, "lead_cap_m": LEAD_CAP}
    for cat, c, paw, h, paw2, h2 in (("pikachu", "p", F["pawP"], F["hP"], pawP2, hP2), ("grey", "g", F["pawG"], F["hG"], pawG2, hG2)):
        n = np.linalg.norm(offs[c], axis=-1)
        for k in ("rec", "walk"):
            b, nb, sb = slide_share(paw, h, track, kind, k)
            a, na, sa = slide_share(paw2, h2, track, kind, k)
            s = stats[c][k]
            down_s = float(downP[kind == k].sum()) / 10
            runs, ends = np.array(s["planted_runs"]), np.array(s["run_ends"])
            res["%s %s" % (cat, k)] = {
                "skating_1.5_to_3cm_before": sb, "skating_1.5_to_3cm_after": sa,
                "skating_readout_after": "planted" if sa < 0.05 else "still skates",
                "M1_over_1.5cm_before": b, "M1_over_1.5cm_after": a, "down_steps_before": nb, "down_steps_after": na,
                "M1_readout_after": "planted" if a <= 0.10 else "slides",
                "added_steps_per_paw_second_down": s["steps"] / max(down_s, 1e-9),
                "planted_stretch_s_median": float(np.median(runs)) / 10 if len(runs) else 0.0,
                "planted_stretch_s_p10_p90": [float(np.percentile(runs, q)) / 10 for q in (10, 90)] if len(runs) else [],
                "waits_share_of_planted_frames": s["waits"] / max(1, s["planted_frames"]),
                "tapping_readout": "no tapping" if len(runs) and np.median(runs) >= 3 else "tapping",
                "planted_stretch_by_end": {why: {"share": float((ends == why).mean()), "median_s": float(np.median(runs[ends == why])) / 10}
                                           for why in ("step", "lift", "end") if (ends == why).any()},
                "motion_down_stretch_s_median": float(np.median(natural_runs(downP, track, kind, k))) / 10,
                "correction_cm_median_down": 100 * float(np.median(n[(kind == k)[:, None] & downP])),
                "correction_cm_p99_down": 100 * float(np.percentile(n[(kind == k)[:, None] & downP], 99))}
    for k in ("rec", "walk"):
        m = (kind == k)[:, None] & downP
        res["grey floating %s" % k] = {"share_over_3cm_before": float((F["hG"][m] > FLOAT).mean()),
                                       "share_over_3cm_after": float((hG2[m & (hP2 <= DOWN)] > FLOAT).mean())}

    # ---- leg weights, and stretch with the steps on sampled frames (as feet_fix.py)
    legP = leg_weights(page.P, page.fp, page.lp)
    legG = leg_weights(page.Q, page.fg, page.lg)
    sk, z = np.load(skel_npz), np.load(uv_npz)
    Qc, own = z["pos"].astype(np.float64), sk["own"].astype(np.float64)
    eP, eG = edges_of(page.fp), edges_of(page.fg)
    rng = np.random.default_rng(1)
    # sampled frames where some paw carries a correction over 2 cm, half recorded, half walks
    big = np.linalg.norm(offs["p"], axis=-1).max(1) > 0.02
    rows = np.r_[rng.choice(np.where((kind == "rec") & big)[0], 40, replace=False),
                 rng.choice(np.where((kind == "walk") & big)[0], 40, replace=False)]
    rows = rows[np.argsort(rows)]
    stP0, stP1, stG = [], [], []
    for i in rows:
        tr = page.tracks[track[i]]
        t = int(F["t"][i])
        T = tr["bones"][t:t + 1].astype(np.float64)
        Hi, dn = tr["H"][t], tr["H"][t][1]
        XP = lbs(page.P, page.ip, page.wp, T[0])
        XG = lbs(page.Q, page.ig, page.wg, region_retarget(T, page.mix, page.C)[0])
        XG += ((XP[::3] @ dn).max() - (XG[::3] @ dn).max()) * dn
        XP1, XG1 = XP.copy(), XG.copy()
        for p in range(4):
            for X1, (v, w, _), c in ((XP1, legP[p], "p"), (XG1, legG[p], "g")):
                X1[v] += w[:, None] * (Hi.T @ offs[c][i, p])[None]
        stP0.append(stretch(page.P, XP, eP))
        stP1.append(stretch(page.P, XP1, eP))
        stG.append(stretch(page.Q, XG1, eG))
    st_own = np.concatenate([stretch(Qc, lbs(Qc, page.ig, page.wg, own[f]), eG) for f in np.linspace(0, len(own) - 1, 40).round().astype(int)])
    stP0, stP1, stG = np.concatenate(stP0), np.concatenate(stP1), np.concatenate(stG)
    ratio = float(np.percentile(stG, 99) / np.percentile(st_own, 99))
    over0, over1 = float((stP0 > np.log(1.3)).mean()), float((stP1 > np.log(1.3)).mean())
    res["stretch"] = {"frames": "80 sampled where a paw carries over 2 cm",
                      "grey_p99_with_steps_over_own_video": ratio,
                      "grey_readout": "moves cleanly" if ratio <= 1.5 else ("some stretching" if ratio <= 3 else "tears"),
                      "pikachu_p99_before": float(np.percentile(stP0, 99)), "pikachu_p99_with_steps": float(np.percentile(stP1, 99)),
                      "pikachu_edges_over_30pct_before": over0, "pikachu_edges_over_30pct_with_steps": over1,
                      "pikachu_readout": "fine" if over1 - over0 <= 0.01 else "stretches"}
    json.dump(res, open(os.path.join(out, "feet_step_check.json"), "w"), indent=1)
    print(json.dumps(res, indent=1), flush=True)

    # ---- for the page: corrections in each cat's own frame (world = H local + root), and the leg weights
    loc = lambda o: np.einsum("nji,npj->npi", H, o)  # noqa: E731  H^T per frame
    oP, oG = loc(offs["p"]).astype(np.float16), loc(offs["g"]).astype(np.float16)
    blobs = [("rec_p", oP[:nrec]), ("walk_p", oP[nrec:]), ("rec_g", oG[:nrec]), ("walk_g", oG[nrec:])]
    counts = {}
    for c, leg in (("p", legP), ("g", legG)):
        blobs.append(("leg_%s_idx" % c, np.concatenate([v for v, _, _ in leg]).astype(np.uint16)))
        blobs.append(("leg_%s_w" % c, np.concatenate([w for _, w, _ in leg]).astype(np.float16)))
        counts[c] = [int(len(v)) for v, _, _ in leg]
    meta, nbytes = pack(os.path.join(out, "feet_step.bin"), blobs)
    json.dump({"blobs": meta, "bytes": nbytes, "paws": PAWS, "leg_counts": counts, "rec_frames": nrec,
               "walk_frames": int(len(kind) - nrec), "units": "metres, the cat's own frame (y down)",
               "from": "paws planted while down (Pikachu's contacts), short lifted steps when a hold would need over "
                       "4 cm; feet_step.py"},
              open(os.path.join(out, "feet_step.json"), "w"))
    print("feet_step: %d bytes" % nbytes, flush=True)

    # ---- picture: walk 3, 0 to 20 s: paw heights and sideways moves, before and after
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
            ax.axhspan(1.5, 3, color="red", alpha=0.08)
            ax.set_ylim(-1, 15)
            ax.set_ylabel("%s\n%s (cm)" % (nm, PAWS[p]))
    axes[0, 0].legend(fontsize=7)
    axes[-1, 0].set_xlabel("walk 3, seconds")
    axes[-1, 1].set_xlabel("walk 3, seconds")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "feet_step.png"), dpi=70)


if __name__ == "__main__":
    main()
