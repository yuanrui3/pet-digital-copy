"""Feet and pose check (work/pets/PROTOCOL_CAT.md, "Feet and pose check"): do the cats' paws slide or float at home,
does the cat sit on the floor, and how closely does the grey and white cat's pose follow Pikachu's?

Everything is posed exactly as the page does it (index.html: poseA2S, poseGrey, setHomeTime): Pikachu skinned with its
own bone motion; the grey cat's bones as blends of Pikachu's (home2_retarget.region_retarget), then the grey cat raised
or lowered so its lowest point (every third vertex, along the room's down) sits where Pikachu's does; both placed in
the room by the track's root place and heading. The room's y points down.
CPU only (numpy, scipy, matplotlib).

usage: python feet_check.py PAGE_DIR OUT_DIR
  PAGE_DIR: the page's data files a2s_cat, a2s_cat2, a2s_walks, paths, room (.bin + .json) and walks.json
Writes OUT_DIR/feet_check.json (measures and read-outs), feet_frames.npz (per-frame paw places, heights and
landmarks, for the fix), feet_worst.png (the five frames whose pose departs most) and feet_timeline.png.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from home2_retarget import region_retarget  # noqa: E402
from home2_transfer import edges_of, read_blobs  # noqa: E402

PAWS = ["paw -x front", "paw +x front", "paw -x hind", "paw +x hind"]
LMS = PAWS + ["nose", "tail tip"]
DOWN = 0.02        # a paw is down when its tip is at most 2 cm above the cat's lowest point
SKID = 0.015       # a step skids when the tip moves more than 1.5 cm sideways in 0.1 s
FLOAT = 0.03       # the grey cat's paw floats when over 3 cm up while Pikachu's is down
NEAR_FLOOR = 0.10  # frames with Pikachu's lowest point within 10 cm of the floor
LAG = 50           # baseline for pose closeness: Pikachu 5 s later (10 frames per second)
CHUNK = 128


def paw_sets(X, e):
    """as home2_retarget.paw_tips, but each paw's 20 lowest vertices (indices), ordered -x front, +x front, -x hind,
    +x hind (Pikachu's frame: y down, z towards the tail)"""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    n = len(X)
    for pct in (90, 88, 85, 82, 78, 74):
        low = X[:, 1] > np.percentile(X[:, 1], pct)
        ee = e[low[e[:, 0]] & low[e[:, 1]]]
        _, lab = connected_components(coo_matrix((np.ones(len(ee)), (ee[:, 0], ee[:, 1])), shape=(n, n)), directed=False)
        labs, cnt = np.unique(lab[low], return_counts=True)
        sets = []
        for c in labs[np.argsort(-cnt)]:
            v = np.where((lab == c) & low)[0]
            if len(v) < 20:
                break
            s = v[np.argsort(-X[v, 1])[:20]]
            if all(np.linalg.norm(X[s].mean(0) - X[u].mean(0)) > 0.04 for u in sets):
                sets.append(s)
        if len(sets) >= 4:
            sets = sorted(sets[:4], key=lambda s: X[s, 2].mean())
            assert X[sets[2], 2].mean() - X[sets[1], 2].mean() >= 0.05, "front and hind paws not clearly apart"
            return sorted(sets[:2], key=lambda s: X[s, 0].mean()) + sorted(sets[2:], key=lambda s: X[s, 0].mean())
    raise AssertionError("no clear four paws")


def landmark_sets(X, faces):
    """the four paws, the nose (20 front-most vertices) and the tail tip (20 highest of the rear fifth)"""
    sets = paw_sets(X, edges_of(faces))
    sets.append(np.argsort(X[:, 2])[:20])
    rear = np.where(X[:, 2] > np.percentile(X[:, 2], 80))[0]
    sets.append(rear[np.argsort(X[rear, 1])[:20]])
    return sets


def skin(T, rest, idx, w):
    """T (n,B,3,4) rest -> frame, rest (V,3), idx/w (V,4) -> (n,V,3), four strongest bones as on the page; written as
    one matrix product (vertex weights times rest coordinates, against all frames' bone matrices)"""
    n, B = T.shape[:2]
    V = len(rest)
    rh = np.c_[rest, np.ones(V)]
    A = np.zeros((V, B * 4))
    for k in range(4):
        for j in range(4):
            np.add.at(A, (np.arange(V), idx[:, k] * 4 + j), w[:, k] * rh[:, j])
    Tm = T.transpose(1, 3, 0, 2).reshape(B * 4, n * 3)  # [b*4+j, f*3+i] = T[f, b, i, j]
    return (A @ Tm).reshape(V, n, 3).transpose(1, 0, 2)


class Page:
    def __init__(self, d):
        def blobs(name):
            m = json.load(open(os.path.join(d, name + ".json")))
            return m, read_blobs(os.path.join(d, name + ".bin"), m)
        am, A = blobs("a2s_cat")
        gm, G = blobs("a2s_cat2")
        wm, W = blobs("a2s_walks")
        pm, Pp = blobs("paths")
        _, Rm = blobs("room")
        self.B = am["bones"]
        self.P, self.ip, self.wp = A["pos"].astype(np.float64), A["skin_idx"].astype(np.int64), A["skin_w"].astype(np.float64)
        self.fp = A["faces"].astype(np.int64)
        self.Q, self.ig, self.wg = G["pos"].astype(np.float64), G["skin_idx"].astype(np.int64), G["skin_w"].astype(np.float64)
        self.fg = G["faces"].astype(np.int64)
        self.mix, self.C = gm["mix"], np.array(gm["centre"], dtype=np.float64)
        rec = A["bones"].reshape(-1, self.B, 3, 4)
        walks = W["bones"].reshape(-1, self.B, 3, 4)
        # the page's floor: 2nd percentile of every tenth room vertex's height (viewer frame, y up)
        ys = np.sort(-Rm["pos"][::10, 1].astype(np.float64))
        self.floor = -ys[int(len(ys) * 0.02)]  # in the room's frame, y down
        root, head = Pp["root"].astype(np.float64), Pp["heading"].astype(np.float64).reshape(-1, 3, 3)
        self.tracks, s = [], 0
        for k, v in enumerate(pm["videos"]):
            n = v["frames"]
            self.tracks.append({"name": "video %d" % (k + 1), "kind": "rec", "bones": rec[s:s + n], "root": root[s:s + n],
                                "H": head[s:s + n]})
            s += n
        runs = json.load(open(os.path.join(d, "walks.json")))["runs"]
        off, nfree = 0, 0
        for k, run in enumerate(runs):
            n = wm["frames_per_run"][k]
            past, pos, rot = np.array(run["past"], float), np.array(run["pos"], float), np.array(run["rot"], float).reshape(-1, 3, 3)
            assert len(past) + len(pos) == n and len(rot) == len(pos), (k, len(past), len(pos), len(rot), n)
            kind = run.get("kind", "free")
            if kind == "free":
                nfree += 1
            self.tracks.append({"name": ("walk %d" % nfree) if kind == "free" else "%s run %d" % (kind, k), "kind": "walk",
                                "sub": kind, "run": k, "bones": walks[off:off + n], "root": np.concatenate([past, pos]),
                                "H": np.concatenate([np.repeat(rot[:1], len(past), 0), rot])})
            off += n
        # vertices the measures need: every third (lowest point, centre) and the landmarks
        self.lp, self.lg = landmark_sets(self.P, self.fp), landmark_sets(self.Q, self.fg)
        self.e3p, self.e3g = np.arange(0, len(self.P), 3), np.arange(0, len(self.Q), 3)
        self.selp = np.concatenate([self.e3p] + self.lp)
        self.selg = np.concatenate([self.e3g] + self.lg)

    def pose(self, T, H):
        """T (n,B,3,4) Pikachu's bones, H (n,3,3) heading -> both cats' selected vertices in the cat's frame, the grey
        cat already raised or lowered as on the page"""
        T = T.astype(np.float64)
        down = H[:, 1, :]
        XP = skin(T, self.P[self.selp], self.ip[self.selp], self.wp[self.selp])
        lowP = np.einsum("nvi,ni->nv", XP[:, :len(self.e3p)], down).max(1)
        XG = skin(region_retarget(T, self.mix, self.C), self.Q[self.selg], self.ig[self.selg], self.wg[self.selg])
        lowG = np.einsum("nvi,ni->nv", XG[:, :len(self.e3g)], down).max(1)
        XG += (lowP - lowG)[:, None, None] * down[:, None, :]
        return XP, XG, lowP

    def split(self, X, e3, sets):
        """-> landmarks (n,6,3) and centre (n,3)"""
        n3, o, L = len(e3), len(e3), []
        for s in sets:
            L.append(X[:, o:o + len(s)].mean(1))
            o += len(s)
        return np.stack(L, 1), X[:, :n3].mean(1)

    def rest_rel(self):
        out = []
        for X, e3, sets in ((self.P, self.e3p, self.lp), (self.Q, self.e3g, self.lg)):
            L = np.stack([X[s].mean(0) for s in sets])
            out.append((L - X[e3].mean(0), np.linalg.norm(L[4] - L[2:4].mean(0))))
        return out


def measure(page):
    rows = {k: [] for k in ("track", "t", "lowW", "pawP", "pawG", "hP", "hG", "relP", "relG")}
    for ti, tr in enumerate(page.tracks):
        n = len(tr["bones"])
        for a in range(0, n, CHUNK):
            b = min(n, a + CHUNK)
            H, r = tr["H"][a:b], tr["root"][a:b]
            XP, XG, lowP = page.pose(tr["bones"][a:b], H)
            LP, cP = page.split(XP, page.e3p, page.lp)
            LG, cG = page.split(XG, page.e3g, page.lg)
            world = lambda L: np.einsum("nij,nlj->nli", H, L) + r[:, None]  # noqa: E731
            pawP, pawG = world(LP[:, :4]), world(LG[:, :4])
            lowW = lowP + r[:, 1]
            rows["track"].append(np.full(b - a, ti))
            rows["t"].append(np.arange(a, b))
            rows["lowW"].append(lowW)
            rows["pawP"].append(pawP)
            rows["pawG"].append(pawG)
            rows["hP"].append(lowW[:, None] - pawP[..., 1])
            rows["hG"].append(lowW[:, None] - pawG[..., 1])
            rows["relP"].append(LP - cP[:, None])
            rows["relG"].append(LG - cG[:, None])
        print("  %-14s %5d frames" % (tr["name"], n), flush=True)
    return {k: np.concatenate(v) for k, v in rows.items()}


def pct(x, q):
    return float(np.percentile(x, q)) if len(x) else float("nan")


def main():
    page_dir, out = sys.argv[1:3]
    os.makedirs(out, exist_ok=True)
    page = Page(page_dir)
    print("Pikachu %d vertices, grey cat %d; %d tracks (%d recorded); floor at %.3f m (room's y down)" % (
        len(page.P), len(page.Q), len(page.tracks), sum(t["kind"] == "rec" for t in page.tracks), page.floor), flush=True)
    for nm, X, sets in (("Pikachu", page.P, page.lp), ("grey", page.Q, page.lg)):
        print("  %s rest landmarks: " % nm + ", ".join("%s %s" % (k, np.round(X[s].mean(0), 3)) for k, s in zip(LMS, sets)), flush=True)
    F = measure(page)
    kind = np.array([page.tracks[i]["kind"] for i in F["track"]])
    np.savez_compressed(os.path.join(out, "feet_frames.npz"), **F, kind=kind, floor=page.floor,
                        lp=np.array(page.lp), lg=np.array(page.lg))
    same = np.r_[F["track"][1:] == F["track"][:-1], False]  # frame t and t+1 on the same track
    res = {"frames": {k: int((kind == k).sum()) for k in ("rec", "walk")}, "floor_m": page.floor}

    # ---- M1 paw slide
    m1 = {}
    for cat, paw, h in (("pikachu", F["pawP"], F["hP"]), ("grey", F["pawG"], F["hG"])):
        down = h <= DOWN
        step = down[:-1] & down[1:] & same[:-1, None]
        skid = np.linalg.norm((paw[1:] - paw[:-1])[..., [0, 2]], axis=-1)
        for k in ("rec", "walk"):
            m = step & (kind[:-1] == k)[:, None]
            s = skid[m]
            m1["%s %s" % (cat, k)] = {"paw_down_share": float(down[kind == k].mean()), "steps": int(m.sum()),
                                      "skid_cm_median": 100 * pct(s, 50), "skid_cm_p90": 100 * pct(s, 90),
                                      "share_over_1.5cm": float((s > SKID).mean()),
                                      "per_paw_share_over_1.5cm": [float((skid[:, p][m[:, p]] > SKID).mean()) for p in range(4)]}
    for k in ("rec", "walk"):
        for cat in ("pikachu", "grey"):
            m1["%s %s" % (cat, k)]["readout"] = "slides" if m1["%s %s" % (cat, k)]["share_over_1.5cm"] > 0.10 else "planted"
        g, p = m1["grey %s" % k]["share_over_1.5cm"], m1["pikachu %s" % k]["share_over_1.5cm"]
        m1["grey vs pikachu %s" % k] = "slides more than Pikachu" if g > 1.25 * p else "no more than Pikachu"
    res["M1_paw_slide"] = m1

    # ---- M2 floating paws
    m2 = {}
    for k in ("rec", "walk"):
        m = (F["hP"] <= DOWN) & (kind == k)[:, None]
        v = F["hG"][m]
        m2[k] = {"pikachu_paw_down": int(m.sum()), "grey_height_cm_median": 100 * pct(v, 50), "grey_height_cm_p90": 100 * pct(v, 90),
                 "share_over_3cm": float((v > FLOAT).mean()),
                 "per_paw_share_over_3cm": [float((F["hG"][:, p][m[:, p]] > FLOAT).mean()) for p in range(4)]}
        m2[k]["readout"] = "lands with Pikachu" if m2[k]["share_over_3cm"] <= 0.15 else "floating paws"
    res["M2_floating_paws"] = m2

    # ---- M3 floor gap (positive: the cat's lowest point above the floor)
    gap = page.floor - F["lowW"]
    m3 = {}
    for k in ("rec", "walk"):
        m = (np.abs(gap) <= NEAR_FLOOR) & (kind == k)
        g = gap[m]
        m3[k] = {"frames_near_floor": int(m.sum()), "share_of_frames": float(m[kind == k].mean()),
                 "gap_cm_median": 100 * pct(g, 50), "gap_cm_p10": 100 * pct(g, 10), "gap_cm_p90": 100 * pct(g, 90),
                 "share_below_floor_2cm_all_frames": float((gap[kind == k] < -0.02).mean())}
        ok = abs(m3[k]["gap_cm_median"]) <= 1.5 and abs(m3[k]["gap_cm_p10"]) <= 3 and abs(m3[k]["gap_cm_p90"]) <= 3
        m3[k]["readout"] = "on the floor" if ok else "floats or sinks"
    res["M3_floor_gap"] = m3

    # ---- M4 pose closeness and M5 range
    (rP, lenP), (rG, lenG) = page.rest_rel()
    kz = lenP / lenG
    dP, dG = F["relP"] - rP, kz * (F["relG"] - rG)
    err = np.linalg.norm(dG - dP, axis=-1)  # (N,6)
    E = err.mean(1)
    idx = np.arange(len(E))
    later = idx + LAG
    ok = later < len(E)
    ok[ok] &= F["track"][later[ok]] == F["track"][idx[ok]]
    Ex = np.full(len(E), np.nan)
    Ex[ok] = np.linalg.norm(dG[ok] - dP[later[ok]], axis=-1).mean(1)
    move = np.linalg.norm(dP, axis=-1).mean(1)
    absE = np.linalg.norm(kz * F["relG"] - F["relP"], axis=-1).mean(1)
    m4 = {"body_length_m_pikachu_grey": [float(lenP), float(lenG)]}
    for k in ("rec", "walk", "all"):
        m = (kind == k) if k != "all" else np.ones(len(E), bool)
        mx = m & ok
        e, b = float(np.median(E[mx])), float(np.median(Ex[mx]))
        m4[k] = {"error_cm_median": 100 * e, "error_cm_p90": 100 * pct(E[m], 90), "baseline_5s_later_cm_median": 100 * b,
                 "pikachu_move_cm_median": 100 * float(np.median(move[m])), "ratio": e / b,
                 "per_landmark_error_cm_median": dict(zip(LMS, (100 * np.median(err[m], 0)).round(2).tolist())),
                 "absolute_shape_difference_cm_median": 100 * float(np.median(absE[m]))}
        m4[k]["readout"] = "follows closely" if e <= b / 3 else ("follows loosely" if e <= 2 * b / 3 else "does not follow")
    res["M4_pose_closeness"] = m4
    m5 = {}
    for k in ("rec", "walk", "all"):
        m = (kind == k) if k != "all" else np.ones(len(E), bool)
        ratio = dG[m].std(0) / np.maximum(dP[m].std(0), 1e-9)  # (6,3)
        med = float(np.median(ratio))
        m5[k] = {"median_ratio": med, "per_landmark": dict(zip(LMS, np.median(ratio, 1).round(3).tolist())),
                 "readout": "softened" if med < 0.8 else ("exaggerated" if med > 1.25 else "same range")}
    res["M5_range"] = m5
    json.dump(res, open(os.path.join(out, "feet_check.json"), "w"), indent=1)
    print(json.dumps(res, indent=1), flush=True)

    # ---- pictures
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    picks = []
    for i in np.argsort(-E):
        if all(F["track"][i] != F["track"][j] or abs(F["t"][i] - F["t"][j]) > 50 for j in picks):
            picks.append(int(i))
        if len(picks) == 5:
            break
    fig, axes = plt.subplots(2, 5, figsize=(21, 8))
    allp, allg = np.arange(0, len(page.P), 2), np.arange(0, len(page.Q), 2)
    for c, i in enumerate(picks):
        tr = page.tracks[F["track"][i]]
        t = int(F["t"][i])
        T = tr["bones"][t:t + 1].astype(np.float64)
        H = tr["H"][t:t + 1]
        XP = skin(T, page.P[allp], page.ip[allp], page.wp[allp])[0]
        Tg = region_retarget(T, page.mix, page.C)
        XG = skin(Tg, page.Q[allg], page.ig[allg], page.wg[allg])[0]
        _, _, lowP = page.pose(T, H)
        down = H[0, 1]
        XG += (lowP[0] - (XG @ down).max()) * down  # close to the page's every-third-vertex match
        for r_, (a, b) in enumerate(((2, 1), (0, 1))):
            ax = axes[r_, c]
            ax.scatter(XP[:, a], XP[:, b], s=0.3, c="tab:orange", alpha=0.3)
            ax.scatter(XG[:, a], XG[:, b], s=0.3, c="tab:blue", alpha=0.3)
            ax.set_aspect("equal")
            ax.invert_yaxis()
            if r_ == 0:
                ax.set_title("%s at %.1f s: pose error %.1f cm" % (tr["name"], t / 10, 100 * E[i]), fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "feet_worst.png"), dpi=70)
    # paw heights and skids on walk 3 (the ship rule's walk), 0 to 20 s
    w3 = [i for i, tr in enumerate(page.tracks) if tr["name"] == "walk 3"][0]
    m = (F["track"] == w3) & (F["t"] < 200)
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    tt = F["t"][m] / 10
    for p in range(4):
        ax = axes[p]
        ax.plot(tt, 100 * F["hP"][m, p], c="tab:orange", label="Pikachu")
        ax.plot(tt, 100 * F["hG"][m, p], c="tab:blue", label="grey cat")
        ax.axhline(100 * DOWN, c="grey", lw=0.5)
        ax.set_ylabel("%s\nheight (cm)" % PAWS[p])
        ax.set_ylim(-1, 20)
    axes[0].legend()
    axes[-1].set_xlabel("walk 3, seconds")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "feet_timeline.png"), dpi=70)
    print("pictures: frames " + ", ".join("%s %.1f s" % (page.tracks[F["track"][i]]["name"], F["t"][i] / 10) for i in picks), flush=True)


if __name__ == "__main__":
    main()
