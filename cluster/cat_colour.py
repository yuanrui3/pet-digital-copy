"""Cat test C1, coat-colour comparison that ignores pose and position, reported without a rule
(work/pets/PROTOCOL_CAT.md, Changes, 2026-11-15 09:30). CPU only.

usage: python cat_colour.py EXP_DIR     (EXP_DIR holds frames.npz, heldout.json and view_baseline.json)
writes EXP_DIR/colour.json
"""
import json
import os
import sys

import cv2
import numpy as np

ERODE = 3  # pixels taken off every mask's edge (256 px crops)
DIRS = 64  # fixed random directions for the sliced distance
Q = np.linspace(0.005, 0.995, 100)  # quantiles compared along each direction
MIN_PIXELS = 200

_rng = np.random.default_rng(0)
_D = _rng.normal(size=(DIRS, 3))
_D /= np.linalg.norm(_D, axis=1, keepdims=True)


def srgb_to_lab(u8):
    """(N,3) uint8 sRGB -> (N,3) CIELAB, D65 white"""
    c = u8.astype(np.float64) / 255
    c = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    m = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]])
    xyz = c @ m.T / np.array([0.95047, 1.0, 1.08883])
    e = 6 / 29
    f = np.where(xyz > e ** 3, np.cbrt(xyz), xyz / (3 * e ** 2) + 4 / 29)
    return np.stack([116 * f[:, 1] - 16, 500 * (f[:, 0] - f[:, 1]), 200 * (f[:, 1] - f[:, 2])], 1)


def cat_pixels(img, mask):
    m = cv2.erode(mask.astype(np.uint8), np.ones((2 * ERODE + 1, 2 * ERODE + 1), np.uint8))
    px = img[m > 0]
    assert len(px) >= MIN_PIXELS, len(px)
    return srgb_to_lab(px)


def mean_diff(p, q):
    return float(np.linalg.norm(p.mean(0) - q.mean(0)))


def spread_dist(p, q):
    return float(np.mean([np.abs(np.quantile(p @ d, Q) - np.quantile(q @ d, Q)).mean() for d in _D]))


def light_chroma(p):
    return float(p[:, 0].mean()), float(np.hypot(p[:, 1], p[:, 2]).mean())


def main(exp):
    fr = np.load(os.path.join(exp, "frames.npz"))
    with open(os.path.join(exp, "heldout.json")) as f:
        h = json.load(f)
    with open(os.path.join(exp, "view_baseline.json")) as f:
        v = json.load(f)
    held, used = fr["held"], fr["used"]
    assert list(held) == h["per_frame"]["frame"] == v["per_frame"]["frame"]
    near = {"nearest_in_time": h["per_frame"]["nearest_used"],
            "nearest_in_viewpoint": v["per_frame"]["viewpoint_nearest"]}

    real = [cat_pixels(fr["real"][i], fr["real_mask"][i]) for i in range(len(fr["real"]))]
    copy = [cat_pixels(fr["drawn"][i], fr["drawn_mask"][i]) for i in range(len(fr["drawn"]))]

    shown = {"copy": [copy[t] for t in held]}
    for k, idx in near.items():
        shown[k] = [real[j] for j in idx]
    truth = [real[t] for t in held]

    per = {m: {"mean_colour_diff": [mean_diff(t, s) for t, s in zip(truth, px)],
               "colour_spread_dist": [spread_dist(t, s) for t, s in zip(truth, px)],
               "lightness_L": [light_chroma(s)[0] for s in px],
               "chroma": [light_chroma(s)[1] for s in px]} for m, px in shown.items()}
    per["real"] = {"lightness_L": [light_chroma(t)[0] for t in truth], "chroma": [light_chroma(t)[1] for t in truth]}
    given = {"mean_colour_diff": float(np.mean([mean_diff(real[u], copy[u]) for u in used])),
             "colour_spread_dist": float(np.mean([spread_dist(real[u], copy[u]) for u in used])),
             "lightness_L": float(np.mean([light_chroma(copy[u])[0] for u in used])),
             "chroma": float(np.mean([light_chroma(copy[u])[1] for u in used])),
             "real_lightness_L": float(np.mean([light_chroma(real[u])[0] for u in used])),
             "real_chroma": float(np.mean([light_chroma(real[u])[1] for u in used]))}

    out = {
        "reported_without_a_rule": "C1's pass rule and verdict are in heldout.json",
        "units": "CIELAB (D65); about 2 = a barely visible difference",
        "pixels": "own mask of each image, %d px taken off its edge" % ERODE,
        "held_back_frames": len(held),
        "held_back_mean": {m: {k: float(np.mean(x)) for k, x in d.items()} for m, d in per.items()},
        "copy_closer_on_frames": {
            k: {meas: int(sum(c < p for c, p in zip(per["copy"][meas], per[k][meas])))
                for meas in ("mean_colour_diff", "colour_spread_dist")} for k in near},
        "given_views_context_copy": given,
        "per_frame": {"frame": [int(t) for t in held],
                      **{"%s_%s" % (m, k): [round(x, 2) for x in d[k]] for m, d in per.items() for k in d}},
    }
    with open(os.path.join(exp, "colour.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: out[k] for k in ("held_back_mean", "copy_closer_on_frames", "given_views_context_copy")},
                     indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
