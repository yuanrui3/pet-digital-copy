"""The body moving with the steps (work/pets/PROTOCOL_CAT.md, "The body moving with the steps"): how real cats' head,
shoulders, hips and tail move over a stride, from the SuperAnimal tracks (sa_pose.py) and the strides found by
gait_stats.py. Over each hind stride (lift to next lift of the same hind paw), in the body's own frame (axis tail base
to neck base, normal pointing down the picture) and in body lengths:
  - head: the nose's place across the axis from the neck base (+ down), minus its stride mean;
  - shoulder, hip: how far the girdle stands above its lower paw, across the axis (the stance leg's extent);
  - tail: the angle of tail base -> tail end against the axis (degrees, + down the picture);
each resampled to BINS phase bins, mean removed, averaged over strides. E1: at least 30 strides; split-half
correlation (odd against even strides) at least 0.5 for a motion to be used. numpy only.
usage: python body_stats.py POSE_DIR GAIT_STATS_JSON OUT_JSON"""
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
FPS = 30.0


def smooth(x, k=5):
    pad = np.r_[np.repeat(x[:1], k // 2), x, np.repeat(x[-1:], k // 2)]
    return np.convolve(pad, np.ones(k) / k, mode="valid")


BINS = 20
pose_dir, stats_json, out = sys.argv[1:4]
strides = [s for s in json.load(open(stats_json))["strides"] if s["leg"].startswith("hind")]
cache, rows = {}, {k: [] for k in ("head", "shoulder", "hip", "tail")}
kept = []
for s in strides:
    clip = s["clip"]
    if clip not in cache:
        z = np.load(os.path.join(pose_dir, clip + ".npz"))
        kp, raw = z["kp"].astype(np.float64), z["raw"]
        a = kp[:, I["neck_base"], :2] - kp[:, I["tail_base"], :2]
        L = np.linalg.norm(a, axis=1)
        ax = a / np.maximum(L, 1e-9)[:, None]
        nrm = np.stack([-ax[:, 1], ax[:, 0]], 1)
        nrm *= np.sign(nrm[:, 1:2] + 1e-9)
        Ls = smooth(L)
        across = lambda p, q: ((kp[:, p, :2] - kp[:, q, :2]) * nrm).sum(1) / Ls  # noqa: E731
        sig = {"head": across(I["nose"], I["neck_base"]),
               "shoulder": np.maximum(across(I["front_left_paw"], I["front_left_thai"]), across(I["front_right_paw"], I["front_right_thai"])),
               "hip": np.maximum(across(I["back_left_paw"], I["back_left_thai"]), across(I["back_right_paw"], I["back_right_thai"]))}
        t = kp[:, I["tail_end"], :2] - kp[:, I["tail_base"], :2]
        sig["tail"] = np.degrees(np.arctan2((t * nrm).sum(1), -(t * ax).sum(1)))  # 0 = straight back along the axis
        conf = {"head": np.minimum(kp[:, I["nose"], 2], kp[:, I["neck_base"], 2]),
                "shoulder": kp[:, [I["front_left_paw"], I["front_right_paw"], I["front_left_thai"], I["front_right_thai"]], 2].min(1),
                "hip": kp[:, [I["back_left_paw"], I["back_right_paw"], I["back_left_thai"], I["back_right_thai"]], 2].min(1),
                "tail": np.minimum(kp[:, I["tail_end"], 2], kp[:, I["tail_base"], 2])}
        cache[clip] = ({k: smooth(v) for k, v in sig.items()}, conf, raw)
    sig, conf, raw = cache[clip]
    s0 = int(round(s["t0"] * FPS)) - int(raw[0])
    s1 = s0 + int(round(s["period_s"] * FPS))
    if s0 < 0 or s1 > len(raw) or s1 - s0 < 8:
        continue
    kept.append(s)
    ph = np.linspace(0, 1, s1 - s0, endpoint=False)
    for k in rows:
        if (conf[k][s0:s1] >= 0.5).all():
            v = np.interp(np.arange(BINS) / BINS, ph, sig[k][s0:s1])
            rows[k].append(v - v.mean())
res = {"hind_strides": len(kept), "bins": BINS, "motions": {}}
for k, v in rows.items():
    V = np.array(v)
    if len(V) < 4:
        res["motions"][k] = {"strides": len(V), "used": False}
        continue
    w = V.mean(0)
    a, b = V[0::2].mean(0), V[1::2].mean(0)
    sh = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else 0.0
    unit = "degrees" if k == "tail" else "body lengths"
    res["motions"][k] = {"strides": len(V), "unit": unit, "waveform": w.round(4).tolist(),
                         "half_peak_to_peak": float((w.max() - w.min()) / 2), "split_half_r": sh,
                         "per_stride_half_ptp_median": float(np.median((V.max(1) - V.min(1)) / 2)),
                         "used": bool(len(V) >= 30 and sh >= 0.5)}
    print("%-8s strides %3d  half peak-to-peak %.4f %s (per stride %.4f)  split-half r %.2f  -> %s" % (
        k, len(V), res["motions"][k]["half_peak_to_peak"], unit, res["motions"][k]["per_stride_half_ptp_median"], sh,
        "used" if res["motions"][k]["used"] else "not used"), flush=True)
json.dump(res, open(out, "w"), indent=1)
