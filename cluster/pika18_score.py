"""The rebuild from this cat's close-up clips, checks R3 and R4 (work/pets/PROTOCOL_CAT.md): the ruler on the rebuilt
cat's own walking clips. Per clip, judge_recon2.py has measured the video and the reconstruction drawn into the video's
pixels (windows of 8 pairs). Here the walking clips are pooled: R3 compares the medians of the leg motion ratio and the
slip ratio; R4 is the classifier's mean probability of being real for the reconstruction's windows, the classifier
trained on the real clips without these clips against version 22 as shown. numpy only.
usage: python pika18_score.py REAL_PAIRS_NPZ V22_NPZ(+...) OUT_JSON NAME:SEQ:RECON2_NPZ [...]   (WALK=comma list of names)"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import judge_score as js  # noqa: E402

real_npz, v22, out = sys.argv[1:4]
items = [a.split(":") for a in sys.argv[4:]]
WALK = os.environ.get("WALK", "0066,0071,0072,0073").split(",")
Xr, Gr, _ = js.real_windows(real_npz)
seqs = list(np.load(real_npz, allow_pickle=True)["seqs"])
Xo, Go, _, _, _ = js.our_windows(v22)
drop = [seqs.index(s) for _, s, _ in items if s in seqs]
keep = ~np.isin(Gr, drop)
model = js.fit_logit(np.r_[Xr[keep], Xo], np.r_[np.ones(keep.sum()), np.zeros(len(Xo))])
res = {"real_windows_used": int(keep.sum()), "version22_mean_p_real": float(js.predict(model, Xo).mean()), "clips": {}}
V, C = [], []
for name, seq, path in items:
    t = np.load(path, allow_pickle=True)
    Xv, Xc = t["video"], t["recon"]
    if not len(Xv):
        res["clips"][name] = {"windows": 0}
        print("%s: no common windows" % name)
        continue
    res["clips"][name] = {"windows": int(len(Xv)), "video_q50": {n: round(float(np.median(Xv[:, k])), 3) for k, n in enumerate(js.FEATS)},
                          "reconstruction_q50": {n: round(float(np.median(Xc[:, k])), 3) for k, n in enumerate(js.FEATS)},
                          "p_real_video": float(js.predict(model, Xv).mean()), "p_real_reconstruction": float(js.predict(model, Xc).mean())}
    c = res["clips"][name]
    print("%-6s windows %2d  leg video %.2f recon %.2f | slip video %.2f recon %.2f | P(real) video %.2f recon %.2f" % (
        name, len(Xv), c["video_q50"]["leg"], c["reconstruction_q50"]["leg"], c["video_q50"]["slip"], c["reconstruction_q50"]["slip"],
        c["p_real_video"], c["p_real_reconstruction"]), flush=True)
    if name in WALK:
        V.append(Xv)
        C.append(Xc)
if V:
    V, C = np.concatenate(V), np.concatenate(C)
    leg_ratio = float(np.median(C[:, 2]) / max(np.median(V[:, 2]), 1e-9))
    slip_diff = float(np.median(C[:, 0]) - np.median(V[:, 0]))
    pv, pc = float(js.predict(model, V).mean()), float(js.predict(model, C).mean())
    res["walking_clips"] = {"windows": int(len(V)), "leg_ratio_recon_over_video": leg_ratio, "slip_recon_minus_video": slip_diff,
                            "p_real_video": pv, "p_real_reconstruction": pc,
                            "R3_passes": bool(1 / 1.5 <= leg_ratio <= 1.5 and abs(slip_diff) <= 0.15), "R4_passes": bool(pc >= 0.40)}
    print("walking clips pooled: %d windows; leg ratio recon/video %.2f, slip recon - video %+.3f -> R3 %s; P(real) video %.2f, reconstruction %.2f (version 22: %.2f) -> R4 %s" % (
        len(V), leg_ratio, slip_diff, "pass" if res["walking_clips"]["R3_passes"] else "FAIL", pv, pc, res["version22_mean_p_real"],
        "pass" if res["walking_clips"]["R4_passes"] else "FAIL"))
else:
    res["walking_clips"] = {"windows": 0, "R3_passes": False, "R4_passes": False}
    print("walking clips: no windows -> R3 and R4 cannot pass")
json.dump(res, open(out, "w"), indent=1)
