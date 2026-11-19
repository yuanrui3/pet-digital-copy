"""The judge without ratings (work/pets/PROTOCOL_CAT.md, "Building the judge"): window measures, the named gait numbers,
the blind test by machine and the trust checks T1 and T3. numpy only.

usage: python judge_score.py REAL_NPZ OUT_JSON NAME=OURS_NPZ [NAME=OURS_NPZ ...]
  the first NAME is version 22 as shown (the classifier is trained on real against it); the others are candidates.
  Candidates with the same windows (judge_ours.py WINDOWS=...) are compared pair by pair when listed as
  PAIRS="a>b,c>d" (a expected to score higher than b)."""
import json
import os
import sys

import numpy as np

NPAIR = int(os.environ.get("NPAIR", 8))   # pairs per window
HOP = int(os.environ.get("HOP", 4))
ASPECT = float(os.environ.get("ASPECT", 1.15))  # the outline at least this much longer than tall
FEATS = ["slip", "slip25", "leg", "torso", "swing_fore", "swing_hind", "mean_fore", "mean_hind", "fore_hind_corr"]
rng = np.random.default_rng(0)


def window_feats(P):
    """P: one window's pairs as a dict of arrays (tx, ty, slip10, slip25, leg, torso, a_fore, a_hind, w, h)"""
    t = np.hypot(P["tx"], P["ty"])
    T = t.sum()
    c = np.corrcoef(P["a_fore"], P["a_hind"])[0, 1] if np.std(P["a_fore"]) > 0 and np.std(P["a_hind"]) > 0 else 0.0
    f = [P["slip10"].sum() / T, P["slip25"].sum() / T, P["leg"].sum() / T, P["torso"].sum() / T,
         P["a_fore"].std() / t.mean(), P["a_hind"].std() / t.mean(), P["a_fore"].mean() / t.mean(),
         P["a_hind"].mean() / t.mean(), c]
    net = np.array([P["tx"].sum(), P["ty"].sum()])
    return np.array(f), {"travel_per_pair": float(t.mean()), "aspect": float(np.median(P["w"] / P["h"])),
                         "slope_deg": float(np.degrees(np.arctan2(abs(net[1]), abs(net[0]) + 1e-12)))}


def real_windows(npz):
    z = np.load(npz, allow_pickle=True)
    R, cols = z["rows"], list(z["cols"])
    c = {n: i for i, n in enumerate(cols)}
    X, G, info = [], [], []
    for s in np.unique(R[:, c["seq"]]):
        S = R[R[:, c["seq"]] == s]
        ok = (S[:, c["ground_how"]] > 0) & np.isfinite(S[:, 4:]).all(1)
        k = 0
        while k + NPAIR <= len(S):
            if ok[k:k + NPAIR].all():
                W = S[k:k + NPAIR]
                P = {n: W[:, c[n]] for n in ("tx", "ty", "slip10", "slip25", "leg", "torso", "a_fore", "a_hind", "w", "h")}
                f, m = window_feats(P)
                m["speed_per_raw_frame"] = float(np.hypot(P["tx"], P["ty"]).sum() / W[:, c["gap"]].sum())
                m["gap"] = float(np.median(W[:, c["gap"]]))
                if m["aspect"] >= ASPECT and m["slope_deg"] <= 30 and m["speed_per_raw_frame"] >= 0.01 and np.isfinite(f).all():
                    X.append(f)
                    G.append(int(s))
                    info.append(m)
                k += HOP
            else:
                k += 1
    return np.array(X), np.array(G), info


def our_windows(npz):
    parts = []
    for j, pth in enumerate(npz.split("+")):  # several files: window numbers offset by 100000 per file
        z = np.load(pth, allow_pickle=True)
        Rj, cols = z["rows"].copy(), list(z["cols"])
        Rj[:, 0] += 100000 * j
        parts.append(Rj)
    R = np.concatenate(parts)
    c = {n: i for i, n in enumerate(cols)}
    X, G, ids, info, kind = [], [], [], [], []
    for wv in np.unique(R[:, c["win"]]):
        W = R[R[:, c["win"]] == wv]
        if len(W) != NPAIR or not np.isfinite(W[:, 5:]).all():
            continue
        P = {n: W[:, c[n]] for n in ("tx", "ty", "slip10", "slip25", "leg", "torso", "a_fore", "a_hind", "w", "h")}
        f, m = window_feats(P)
        m["speed_per_raw_frame"] = float(np.hypot(P["tx"], P["ty"]).sum() / (W[:, c["gap"]].sum() * 3))
        m["gap"] = float(W[0, c["gap"]])
        if m["aspect"] >= ASPECT and m["slope_deg"] <= 30 and m["speed_per_raw_frame"] >= 0.01 and np.isfinite(f).all():
            X.append(f)
            G.append(int(W[0, c["track"]]))
            ids.append(int(wv))
            info.append(m)
            kind.append(int(W[0, c["kind"]]))
    return np.array(X), np.array(G), np.array(ids), info, np.array(kind)


def fit_logit(X, y, l2=1.0, iters=60):
    """standardised, L2-regularised logistic regression by Newton steps; classes weighted equally"""
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = np.c_[(X - mu) / sd, np.ones(len(X))]
    wts = np.where(y == 1, 0.5 / max(y.sum(), 1), 0.5 / max((1 - y).sum(), 1)) * len(y)
    b = np.zeros(Z.shape[1])
    reg = l2 * np.r_[np.ones(Z.shape[1] - 1), 0]
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(Z @ b, -30, 30)))
        g = Z.T @ (wts * (p - y)) + reg * b
        Hm = (Z * (wts * p * (1 - p))[:, None]).T @ Z + np.diag(reg) + 1e-8 * np.eye(Z.shape[1])
        step = np.linalg.solve(Hm, g)
        b -= step
        if np.abs(step).max() < 1e-8:
            break
    return {"mu": mu, "sd": sd, "b": b}


def predict(m, X):
    Z = np.c_[(X - m["mu"]) / m["sd"], np.ones(len(X))]
    return 1 / (1 + np.exp(-np.clip(Z @ m["b"], -30, 30)))


def auc(p, y):
    o = np.argsort(p)
    r = np.empty(len(p))
    r[o] = np.arange(1, len(p) + 1)
    n1, n0 = y.sum(), (1 - y).sum()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / max(n1 * n0, 1))


def grouped_cv(X1, G1, X0, G0, folds=5, seed=0):
    """class 1 against class 0, folds by group; -> held-out balanced accuracy, AUC, and each window's held-out p"""
    r = np.random.default_rng(seed)
    f1 = {g: k % folds for k, g in enumerate(r.permutation(np.unique(G1)))}
    f0 = {g: k % folds for k, g in enumerate(r.permutation(np.unique(G0)))}
    F1, F0 = np.array([f1[g] for g in G1]), np.array([f0[g] for g in G0])
    p1, p0 = np.zeros(len(X1)), np.zeros(len(X0))
    for k in range(folds):
        Xt = np.r_[X1[F1 != k], X0[F0 != k]]
        yt = np.r_[np.ones((F1 != k).sum()), np.zeros((F0 != k).sum())]
        m = fit_logit(Xt, yt)
        p1[F1 == k], p0[F0 == k] = predict(m, X1[F1 == k]), predict(m, X0[F0 == k])
    bal = 0.5 * ((p1 > 0.5).mean() + (p0 <= 0.5).mean())
    return float(bal), auc(np.r_[p1, p0], np.r_[np.ones(len(p1)), np.zeros(len(p0))]), p1, p0


def q(x):
    return [round(float(v), 3) for v in np.percentile(x, [25, 50, 75])]


def main():
    real_npz, out = sys.argv[1:3]
    cands = [a.split("=", 1) for a in sys.argv[3:]]
    Xr, Gr, ir = real_windows(real_npz)
    res = {"real": {"windows": len(Xr), "clips": int(len(np.unique(Gr))),
                    "travel_per_pair_median": float(np.median([m["travel_per_pair"] for m in ir])),
                    "speed_per_raw_frame_median": float(np.median([m["speed_per_raw_frame"] for m in ir])),
                    "kept_gap_median": float(np.median([m["gap"] for m in ir])),
                    "measures_q25_q50_q75": {n: q(Xr[:, k]) for k, n in enumerate(FEATS)}}}
    print("real: %d windows from %d clips; travel per pair median %.3f sizes" % (
        len(Xr), len(np.unique(Gr)), res["real"]["travel_per_pair_median"]), flush=True)
    # T1: real against real
    t1 = []
    for s in range(20):
        r = np.random.default_rng(100 + s)
        gs = r.permutation(np.unique(Gr))
        a = np.isin(Gr, gs[:len(gs) // 2])
        if a.sum() < 10 or (~a).sum() < 10:
            continue
        t1.append(grouped_cv(Xr[a], Gr[a], Xr[~a], Gr[~a], seed=s)[0])
    res["T1_real_vs_real"] = {"balanced_accuracy_mean": float(np.mean(t1)), "max": float(np.max(t1)), "splits": len(t1),
                              "passes": bool(np.mean(t1) <= 0.60)}
    print("T1 real against real: balanced accuracy mean %.3f (max %.3f): %s" % (np.mean(t1), np.max(t1),
          "pass" if np.mean(t1) <= 0.60 else "FAIL"), flush=True)
    ours = {}
    for name, path in cands:
        X, G, ids, info, kind = our_windows(path)
        ours[name] = {"X": X, "G": G, "ids": ids, "kind": kind}
        res[name] = {"windows": len(X), "tracks": int(len(np.unique(G))),
                     "travel_per_pair_median": float(np.median([m["travel_per_pair"] for m in info])) if info else None,
                     "speed_per_raw_frame_median": float(np.median([m["speed_per_raw_frame"] for m in info])) if info else None,
                     "measures_q25_q50_q75": {n: q(X[:, k]) for k, n in enumerate(FEATS)} if len(X) else None}
    # the blind test by machine: real against the first candidate (version 22 as shown)
    base = cands[0][0]
    Xo, Go = ours[base]["X"], ours[base]["G"]
    bal, a, p1, p0 = grouped_cv(Xr, Gr, Xo, Go)
    res["machine_blind_test"] = {"against": base, "balanced_accuracy": bal, "auc": a,
                                 "real_windows_called_real": float((p1 > 0.5).mean()), "ours_called_real": float((p0 > 0.5).mean())}
    for kd, nm in ((1, "recordings"), (2, "walks")):
        mk = ours[base]["kind"] == kd
        if mk.sum() >= 10:
            b2, a2, _, q0 = grouped_cv(Xr, Gr, Xo[mk], Go[mk])
            res["machine_blind_test"][nm] = {"balanced_accuracy": b2, "auc": a2, "windows": int(mk.sum())}
    print("machine blind test, real against %s: balanced accuracy %.3f, AUC %.3f" % (base, bal, a), flush=True)
    model = fit_logit(np.r_[Xr, Xo], np.r_[np.ones(len(Xr)), np.zeros(len(Xo))])
    res["model_weights_standardised"] = {n: round(float(b), 3) for n, b in zip(FEATS, model["b"][:-1])}
    for name in ours:
        if len(ours[name]["X"]):
            ours[name]["p"] = predict(model, ours[name]["X"])
            res[name]["mean_p_real"] = float(ours[name]["p"].mean())
    res["real"]["mean_p_real_heldout"] = float(p1.mean())
    # paired comparisons
    res["pairs"] = {}
    for pr in [x for x in os.environ.get("PAIRS", "").split(",") if x]:
        hi, lo = pr.split(">")
        A, B = ours[hi], ours[lo]
        common = np.intersect1d(A["ids"], B["ids"])
        ia = {w: k for k, w in enumerate(A["ids"])}
        ib = {w: k for k, w in enumerate(B["ids"])}
        d = np.array([A["p"][ia[w]] - B["p"][ib[w]] for w in common])
        gg = np.array([A["G"][ia[w]] for w in common])
        boots = []
        ug = np.unique(gg)
        for _ in range(2000):
            pick = rng.choice(ug, len(ug))
            boots.append(np.concatenate([d[gg == g_] for g_ in pick]).mean())
        lo5, hi95 = np.percentile(boots, [5, 95])
        fa = {n: float(np.median(A["X"][[ia[w] for w in common], k]) - np.median(B["X"][[ib[w] for w in common], k])) for k, n in enumerate(FEATS[:4])}
        res["pairs"][pr] = {"windows": int(len(common)), "mean_difference_p_real": float(d.mean()), "interval_90": [float(lo5), float(hi95)],
                            "median_measure_differences": fa}
        print("pair %s: %d windows, P(real) difference %.3f [%.3f, %.3f]" % (pr, len(common), d.mean(), lo5, hi95), flush=True)
    # check T2: each clip's video windows and our reconstruction of the same clip, scored by a model trained without it
    res["T2"] = {}
    zs = np.load(real_npz, allow_pickle=True)
    seq_names = list(zs["seqs"])
    for item in [x for x in os.environ.get("T2", "").split(",") if x]:
        name, seq, path = item.split(":")
        t = np.load(path, allow_pickle=True)
        Xv, Xc = t["video"], t["recon"]
        if not len(Xv):
            res["T2"][name] = {"windows": 0, "passes": False}
            print("T2 %s: no common windows" % name, flush=True)
            continue
        keep = Gr != (seq_names.index(seq) if seq in seq_names else -1)
        m2 = fit_logit(np.r_[Xr[keep], Xo], np.r_[np.ones(keep.sum()), np.zeros(len(Xo))])
        pv, pc, po = float(predict(m2, Xv).mean()), float(predict(m2, Xc).mean()), float(predict(m2, Xo).mean())
        leg_ratio = float(np.median(Xc[:, 2]) / max(np.median(Xv[:, 2]), 1e-9))
        slip_diff = float(np.median(Xc[:, 0]) - np.median(Xv[:, 0]))
        ok1 = (1 / 1.5 <= leg_ratio <= 1.5) and abs(slip_diff) <= 0.15
        ok2 = abs(pc - pv) < abs(pc - po)
        res["T2"][name] = {"windows": int(len(Xv)), "video_q50": {n: round(float(np.median(Xv[:, k])), 3) for k, n in enumerate(FEATS)},
                           "reconstruction_q50": {n: round(float(np.median(Xc[:, k])), 3) for k, n in enumerate(FEATS)},
                           "leg_ratio_recon_over_video": leg_ratio, "slip_recon_minus_video": slip_diff,
                           "p_real_video": pv, "p_real_reconstruction": pc, "p_real_version22": po,
                           "passes_i_measures": bool(ok1), "passes_ii_score": bool(ok2), "passes": bool(ok1 and ok2)}
        print("T2 %s: %d windows; leg ratio recon/video %.2f, slip recon - video %+.3f (%s); P(real) video %.2f, reconstruction %.2f, version 22 %.2f (%s)" % (
            name, len(Xv), leg_ratio, slip_diff, "pass" if ok1 else "FAIL", pv, pc, po, "pass" if ok2 else "FAIL"), flush=True)
    json.dump(res, open(out, "w"), indent=1)
    print("\n%-16s %s" % ("measure (q50)", "  ".join("%-12s" % n[:12] for n in ["real"] + list(ours))))
    for k, n in enumerate(FEATS):
        print("%-16s %s" % (n, "  ".join("%-12.3f" % np.median(v) for v in [Xr[:, k]] + [ours[c]["X"][:, k] for c in ours if len(ours[c]["X"])])))


if __name__ == "__main__":
    main()
