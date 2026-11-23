"""The video-model judge (work/pets/PROTOCOL_CAT.md, "A video-model judge"): scores and the checks J0 to J3.

Score of a set of our clips: the unbiased squared MMD with the cubic polynomial kernel (k(x, y) = (x.y / 1024 + 1)^3,
as KID and JEDi) between its V-JEPA 2 features and the real clips'; lower is more like the real cats. A against B:
the difference of the two scores on the same windows. Uncertainty: 2,000 bootstrap draws, resampling our windows
(the same draw for every version: paired) and the real clips by source video (all of a video's windows together);
pairs drawn from the same original clip are left out of the within-set means.
numpy only.
usage: python jepa_score.py DIR OUT_JSON
  DIR holds real_feat.npy + real.npz (clip names) and <name>_feat.npy + <name>.npz (win) for v22, g5, g9, g12 and the
  ladder: slide4, slide8, slide16, stiff50, stiff100, hold6, hold12, jitter1, jitter3, jitter6, back."""
import json
import os
import sys

import numpy as np

d, out = sys.argv[1:3]
NB, SEED = 2000, 7
LADDER = {"paws sliding": ["slide4", "slide8", "slide16"], "stiff legs": ["stiff50", "stiff100"],
          "held poses": ["hold6", "hold12"], "bone jitter": ["jitter1", "jitter3", "jitter6"], "played backwards": ["back"]}
VERS = ["v22", "g5", "g9", "g12"]


def kern(X, Y):
    return (X @ Y.T / X.shape[1] + 1.0) ** 3


R = np.load(os.path.join(d, "real_feat.npy")).astype(np.float64)
rclip = np.load(os.path.join(d, "real.npz"))["clip"]
videos = sorted(set(rclip.tolist()))
rv = np.array([videos.index(c) for c in rclip])
names = VERS + [n for v in LADDER.values() for n in v]
F, W = {}, {}
for n in names:
    p = os.path.join(d, n + "_feat.npy")
    if os.path.exists(p):
        F[n] = np.load(p).astype(np.float64)
        W[n] = np.load(os.path.join(d, n + ".npz"))["win"]
common = sorted(set.intersection(*[set(w.tolist()) for w in W.values()]))
for n in F:  # the same windows, in the same order, for every set
    pos = {w: i for i, w in enumerate(W[n].tolist())}
    F[n] = F[n][[pos[w] for w in common]]
nw = len(common)
KRR = kern(R, R)
KR = {n: kern(F[n], R) for n in F}
KS = {n: kern(F[n], F[n]) for n in F}


def within(K, c):
    """mean of K over pairs of the draw (multiplicities c), pairs from the same original left out"""
    s = c @ K @ c - (c * c * np.diag(K)).sum()
    return s / (c.sum() ** 2 - (c * c).sum())


def mmd(n, cw, cr):
    return within(KS[n], cw) + within(KRR, cr) - 2 * (cw @ KR[n] @ cr) / (cw.sum() * cr.sum())


rng = np.random.default_rng(SEED)
ones_w, ones_r = np.ones(nw), np.ones(len(R))
draws = []
for _ in range(NB):
    cw = np.bincount(rng.integers(0, nw, nw), minlength=nw).astype(np.float64)
    pick = rng.integers(0, len(videos), len(videos))
    cr = np.bincount(pick, minlength=len(videos))[rv].astype(np.float64)
    draws.append((cw, cr))
score = {n: mmd(n, ones_w, ones_r) for n in F}
boot = {n: np.array([mmd(n, cw, cr) for cw, cr in draws]) for n in F}


def compare(a, b):
    """a's score minus b's (negative: a is more like the real cats), with the 95% interval"""
    x = boot[a] - boot[b]
    return {"diff": score[a] - score[b], "lo": float(np.percentile(x, 2.5)), "hi": float(np.percentile(x, 97.5))}


res = {"windows": nw, "real_clips": int(len(R)), "real_videos": len(videos), "scores": score}
# J1: the user's three verdicts
v1, v2, v3 = compare("g5", "v22"), compare("g12", "g5"), compare("g9", "g5")
gain = lambda c: -c["diff"]  # noqa: E731
j1 = {"V1 walking cycle better than version 22": dict(v1, ok=bool(v1["hi"] < 0)),
      "V2 head fix better than version 24": dict(v2, ok=bool(v2["hi"] < 0)),
      "V3 levelling not clearly better than version 23": dict(v3, ok=bool(v3["hi"] >= 0 or gain(v3) < min(gain(v1), gain(v2))))}
res["J1"] = {"checks": j1, "passes": all(c["ok"] for c in j1.values())}
# J2: the defect ladder on g12
j2 = {}
for kind, steps in LADDER.items():
    if not all(s in F for s in steps):
        continue
    seq = [score["g12"]] + [score[s] for s in steps]
    top = compare(steps[-1], "g12")
    j2[kind] = {"scores": seq, "worse_each_step": bool(all(b > a for a, b in zip(seq[:-1], seq[1:]))),
                "strongest_against_g12": top, "ok": bool(all(b > a for a, b in zip(seq[:-1], seq[1:])) and top["lo"] > 0)}
res["J2"] = {"kinds": j2, "kinds_passing": sum(v["ok"] for v in j2.values()),
             "passes": sum(v["ok"] for v in j2.values()) >= 4}
# J3: held-out real clips (half the videos, alternate in name order) against the other half, and each version
half_a = np.isin(rv, np.arange(0, len(videos), 2))
A, B = R[half_a], R[~half_a]
va, vb = rv[half_a], rv[~half_a]
KAA, KBB, KAB = kern(A, A), kern(B, B), kern(A, B)
KVB = {n: kern(F[n], B) for n in VERS if n in F}


def grp(vids, pick):
    u = sorted(set(vids.tolist()))
    c = np.bincount(np.array(u)[pick], minlength=len(videos))
    return c[vids].astype(np.float64)


ua, ub = sorted(set(va.tolist())), sorted(set(vb.tolist()))
rr, rver = [], {n: [] for n in KVB}
for b in range(NB):
    cw = draws[b][0]
    ca = grp(va, rng.integers(0, len(ua), len(ua)))
    cb = grp(vb, rng.integers(0, len(ub), len(ub)))
    rr.append(within(KAA, ca) + within(KBB, cb) - 2 * (ca @ KAB @ cb) / (ca.sum() * cb.sum()))
    for n in KVB:
        rver[n].append(within(KS[n], cw) + within(KBB, cb) - 2 * (cw @ KVB[n] @ cb) / (cw.sum() * cb.sum()))
o_a, o_b, o_w = np.ones(len(A)), np.ones(len(B)), ones_w
rr0 = within(KAA, o_a) + within(KBB, o_b) - 2 * (o_a @ KAB @ o_b) / (len(A) * len(B))
j3 = {"real_half_against_other_half": rr0}
for n in KVB:
    s0 = within(KS[n], o_w) + within(KBB, o_b) - 2 * (o_w @ KVB[n] @ o_b) / (nw * len(B))
    x = np.array(rr) - np.array(rver[n])
    j3[n] = {"score_against_half": s0, "diff_real_minus_ours": rr0 - s0, "lo": float(np.percentile(x, 2.5)),
             "hi": float(np.percentile(x, 97.5)), "ok": bool(np.percentile(x, 97.5) < 0)}
res["J3"] = {"checks": j3, "real_videos_each_half": [len(ua), len(ub)], "real_clips_each_half": [int(len(A)), int(len(B))],
             "passes": all(j3[n]["ok"] for n in KVB)}
res["J0"] = {"real_clips": int(len(R)), "passes": bool(len(R) >= 60)}
json.dump(res, open(out, "w"), indent=1)
print("windows %d, real clips %d from %d videos" % (nw, len(R), len(videos)))
for n in F:
    print("  %-9s score %.4f" % (n, score[n]))
for k, c in j1.items():
    print("J1 %-48s diff %+.4f [%+.4f, %+.4f] %s" % (k, c["diff"], c["lo"], c["hi"], "ok" if c["ok"] else "NO"))
for k, c in j2.items():
    print("J2 %-17s %s  strongest %+.4f [%+.4f, %+.4f] %s" % (k, " < ".join("%.4f" % s for s in c["scores"]),
          c["strongest_against_g12"]["diff"], c["strongest_against_g12"]["lo"], c["strongest_against_g12"]["hi"], "ok" if c["ok"] else "NO"))
for n in KVB:
    c = j3[n]
    print("J3 real half %.4f vs %-4s %.4f: diff %+.4f [%+.4f, %+.4f] %s" % (rr0, n, c["score_against_half"], c["diff_real_minus_ours"],
                                                                           c["lo"], c["hi"], "ok" if c["ok"] else "NO"))
print("J0 %s  J1 %s  J2 %s (%d of %d kinds)  J3 %s" % (res["J0"]["passes"], res["J1"]["passes"], res["J2"]["passes"],
                                                      res["J2"]["kinds_passing"], len(j2), res["J3"]["passes"]))
