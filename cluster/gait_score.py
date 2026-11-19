"""The walking cycle's check G1 (work/pets/PROTOCOL_CAT.md, "A built-in walking cycle on the copy's walks"): the ruler
on the 16 free walks, drawn side-on on version 22's own windows and cameras. Medians of the leg motion and slip
measures for the real clips, version 22's walks and the walking cycle; and, for information, the classifier's mean
probability of being real (trained on the real clips against version 22's walks as shown). numpy only.
usage: python gait_score.py REAL_PAIRS_NPZ V22_WALK_NPZ GAIT_NPZ OUT_JSON"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import judge_score as js  # noqa: E402

real, v22, gait, out = sys.argv[1:5]
Xr, Gr, _ = js.real_windows(real)
Xv = js.our_windows(v22)[0]
Xg = js.our_windows(gait)[0]
model = js.fit_logit(np.r_[Xr, Xv], np.r_[np.ones(len(Xr)), np.zeros(len(Xv))])
iL, iS = js.FEATS.index("leg"), js.FEATS.index("slip")
res = {}
for name, X in (("real clips", Xr), ("version 22 walks", Xv), ("walking cycle", Xg)):
    res[name] = {"windows": int(len(X)), "leg_median": float(np.median(X[:, iL])), "slip_median": float(np.median(X[:, iS])),
                 "p_real_mean": float(js.predict(model, X).mean())}
    print("%-17s windows %3d  leg %.2f  slip %.2f  P(real) %.2f" % (name, len(X), res[name]["leg_median"], res[name]["slip_median"],
                                                                   res[name]["p_real_mean"]), flush=True)
g = res["walking cycle"]
res["G1_passes"] = bool(g["leg_median"] >= 0.45 and g["slip_median"] <= 0.25)
print("G1: leg %.2f (needs 0.45 or more), slip %.2f (needs 0.25 or less) -> %s" % (g["leg_median"], g["slip_median"],
                                                                                   "pass" if res["G1_passes"] else "FAIL"))
json.dump(res, open(out, "w"), indent=1)
