"""Real cats' gait from video (work/pets/PROTOCOL_CAT.md, "Real cats' gait from video"), step 2: walking measures from
the SuperAnimal keypoints (sa_pose.py), 30 frames a second, in body lengths (neck base to tail base) so the camera's
distance does not matter.
  - frames used: side-on (body axis within 35 degrees of level and at least 0.7 of the clip's 90th-percentile length),
    with confident shoulder or hip and paw points (score >= 0.5);
  - per leg: u = the paw's fore-aft place from its shoulder or hip along the body axis (+ towards the head), h = its
    place across the axis (+ downwards), both over body length, smoothed over 5 frames; the paw is in swing while u
    grows faster than SWING_V body lengths a second (moving forward against the body) for 3 frames or more, else in
    stance (on the ground, so the body passes over it: its backward speed against the body is the walking speed);
  - per stride (swing start to next swing start, 0.3 to 2.5 s, all frames usable): period, swing and stance times,
    duty factor, stance speed (body lengths a second), sweep (u at touch-down minus u at lift-off), where it lands
    and lifts (u), lift height (stance h minus the least h in swing), and the lag of each front leg's swing after the
    same side's hind leg's, as a share of the stride.
E1 (half of it): stride frequency and duty factor against the published ranges for walking cats. numpy only.
usage: python gait_stats.py POSE_DIR OUT_JSON"""
import glob
import json
import os
import sys

import numpy as np

NAMES = ["nose", "upper_jaw", "lower_jaw", "mouth_end_right", "mouth_end_left", "right_eye", "right_earbase", "right_earend",
         "right_antler_base", "right_antler_end", "left_eye", "left_earbase", "left_earend", "left_antler_base",
         "left_antler_end", "neck_base", "neck_end", "throat_base", "throat_end", "back_base", "back_end", "back_middle",
         "tail_base", "tail_end", "front_left_thai", "front_left_knee", "front_left_paw", "front_right_thai",
         "front_right_knee", "front_right_paw", "back_left_paw", "back_left_thai", "back_right_thai", "back_left_knee",
         "back_right_knee", "back_right_paw", "belly_bottom", "body_middle_right", "body_middle_left"]
I = {n: k for k, n in enumerate(NAMES)}
LEGS = {"front left": ("front_left_thai", "front_left_paw"), "front right": ("front_right_thai", "front_right_paw"),
        "hind left": ("back_left_thai", "back_left_paw"), "hind right": ("back_right_thai", "back_right_paw")}
FPS, SWING_V, SMOOTH = 30.0, 0.25, 5

pose_dir, out = sys.argv[1:3]


def smooth(x, k=SMOOTH):
    pad = np.r_[np.repeat(x[:1], k // 2), x, np.repeat(x[-1:], k // 2)]
    return np.convolve(pad, np.ones(k) / k, mode="valid")


def runs(mask):
    """start, end (exclusive) of each run of True"""
    d = np.diff(np.r_[0, mask.astype(int), 0])
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


strides, lags = [], []
for path in sorted(glob.glob(os.path.join(pose_dir, "cat-*.npz"))):
    clip = os.path.basename(path)[:-4]
    if clip[-4:] in ("0078", "0079", "0080"):
        continue
    z = np.load(path)
    kp, raw = z["kp"].astype(np.float64), z["raw"]
    if len(raw) < 20 or not (np.diff(raw) == 1).all():
        continue
    a = kp[:, I["neck_base"], :2] - kp[:, I["tail_base"], :2]
    L = np.linalg.norm(a, axis=1)
    ax = a / np.maximum(L, 1e-9)[:, None]
    nrm = np.stack([-ax[:, 1], ax[:, 0]], 1)
    nrm *= np.sign(nrm[:, 1:2] + 1e-9)  # pointing down the picture
    level = np.degrees(np.arcsin(np.abs(ax[:, 1]))) <= 35
    side = level & (L >= 0.7 * np.percentile(L, 90)) & (kp[:, I["neck_base"], 2] >= 0.5) & (kp[:, I["tail_base"], 2] >= 0.5)
    Ls = smooth(L)
    onsets = {}
    for leg, (gn, pn) in LEGS.items():
        g, p = kp[:, I[gn]], kp[:, I[pn]]
        ok = side & (g[:, 2] >= 0.5) & (p[:, 2] >= 0.5)
        d = p[:, :2] - g[:, :2]
        u = smooth((d * ax).sum(1) / Ls)
        h = smooth((d * nrm).sum(1) / Ls)
        du = np.gradient(u) * FPS
        swing = du > SWING_V
        sw = [(s, e) for s, e in runs(swing) if e - s >= 3]
        onsets[leg] = []
        for (s0, e0), (s1, _) in zip(sw[:-1], sw[1:]):
            T = (s1 - s0) / FPS
            if not (0.3 <= T <= 2.5) or not ok[s0:s1].all():
                continue
            st = slice(e0, s1)  # stance: from touch-down (end of swing) to the next lift-off
            v = float(-du[st].mean())
            if v <= 0:
                continue
            stride = {"clip": clip, "leg": leg, "t0": float(raw[s0] / FPS), "period_s": T, "swing_s": (e0 - s0) / FPS,
                      "stance_s": (s1 - e0) / FPS, "duty": (s1 - e0) / (s1 - s0), "speed_bl_s": v,
                      "sweep_bl": float(u[e0] - u[s1 - 1]), "land_u_bl": float(u[e0]), "lift_u_bl": float(u[s1 - 1]),
                      "lift_h_bl": float(np.median(h[st]) - h[s0:e0].min()), "body_px": float(np.median(L[s0:s1]))}
            strides.append(stride)
            onsets[leg].append((s0, s1))
    # lag of each front swing after the same side's hind swing, as a share of that hind stride
    for sideword in ("left", "right"):
        for (hs, he) in onsets.get("hind " + sideword, []):
            fs = [s for s, _ in onsets.get("front " + sideword, []) if hs <= s < he]
            if fs:
                lags.append({"clip": clip, "side": sideword, "lag": (fs[0] - hs) / (he - hs)})

S = {k: np.array([s[k] for s in strides]) for k in ("period_s", "swing_s", "stance_s", "duty", "speed_bl_s", "sweep_bl",
                                                     "land_u_bl", "lift_u_bl", "lift_h_bl")}
q = lambda x: np.percentile(x, [25, 50, 75]).round(3).tolist() if len(x) else None  # noqa: E731
res = {"strides": len(strides), "clips": sorted(set(s["clip"] for s in strides)),
       "per_leg_strides": {leg: sum(s["leg"] == leg for s in strides) for leg in LEGS},
       "q25_q50_q75": {k: q(v) for k, v in S.items()},
       "stride_frequency_hz_q50": float(np.median(1 / S["period_s"])) if len(strides) else None,
       "front_after_hind_lag_q25_q50_q75": q(np.array([x["lag"] for x in lags])), "lag_count": len(lags)}
# by speed: thirds of the measured speeds
if len(strides) >= 9:
    edges = np.percentile(S["speed_bl_s"], [0, 33.3, 66.7, 100])
    res["by_speed"] = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (S["speed_bl_s"] >= lo) & (S["speed_bl_s"] <= hi)
        res["by_speed"].append({"speed_bl_s": [round(float(lo), 2), round(float(hi), 2)], "n": int(m.sum()),
                                **{k: round(float(np.median(v[m])), 3) for k, v in S.items()}})
f, dty = res["stride_frequency_hz_q50"], (res["q25_q50_q75"]["duty"] or [0, 0, 0])[1]
res["E1_numbers"] = {"stride_frequency_hz": f, "duty": dty,
                     "passes": bool(f is not None and 0.6 <= f <= 2.0 and 0.5 <= dty <= 0.8)}
json.dump({"summary": res, "strides": strides, "lags": lags}, open(out, "w"), indent=1)
print(json.dumps(res, indent=1))
