"""The judge without ratings (work/pets/PROTOCOL_CAT.md, "Building the judge"): the pair measures of the real cat clips
(Lab4D's public cat-85 set, as preprocessed in its archive: outlines, image motion between consecutive kept frames).

Per clip: Annotations/crop-256.npy (outline and validity, the cat's box resized to 256 px), its crop2raw (raw = f * crop +
p per axis), FlowFW_1/crop-256.npy (motion in crop pixels of the next frame's box) and the full-frame versions (the
ground under the feet). Raw image motion of an outline pixel: raw(next box, crop + flow) - raw(this box, crop).
CPU only.
usage: python judge_real.py REAL_DIR OUT_NPZ
  env ONLY=seq1,seq2: only these clips; BORDER_OK=1: also pairs whose outline touches the frame's edge (check T2)
  REAL_DIR: the unpacked database/processed/... folders and kept_frames.json (each clip's kept frames' raw numbers)"""
import json
import os
import sys

import numpy as np
from scipy.ndimage import binary_dilation

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from judge_feat import NAMES, pair_measures  # noqa: E402

real, out = sys.argv[1:3]
base = os.path.join(real, "database", "processed")
kept = json.load(open(os.path.join(real, "kept_frames.json")))
seqs = sorted(os.listdir(os.path.join(base, "Annotations", "Full-Resolution")))
ONLY = [x for x in os.environ.get("ONLY", "").split(",") if x]
BORDER_OK = os.environ.get("BORDER_OK") == "1"
rows, meta = [], []
uu, vv = np.meshgrid(np.arange(256, dtype=np.float64), np.arange(256, dtype=np.float64))
for si, seq in enumerate(seqs):
    A = os.path.join(base, "Annotations", "Full-Resolution", seq)
    Fd = os.path.join(base, "FlowFW_1", "Full-Resolution", seq)
    if not os.path.exists(os.path.join(Fd, "crop-256.npy")) or seq not in kept or (ONLY and seq not in ONLY):
        print("skip", seq)
        continue
    M, Mf = np.load(os.path.join(A, "crop-256.npy")), np.load(os.path.join(A, "full-256.npy"))
    K, Kf = np.load(os.path.join(A, "crop-256-crop2raw.npy")), np.load(os.path.join(A, "full-256-crop2raw.npy"))
    det = np.load(os.path.join(A, "crop-256-is_detected.npy"))
    Fc, Ff = np.load(os.path.join(Fd, "crop-256.npy")), np.load(os.path.join(Fd, "full-256.npy"))
    raw_ids = kept[seq]
    if len(raw_ids) != len(M):
        print("skip %s: %d frame names for %d frames" % (seq, len(raw_ids), len(M)))
        continue
    nv = 0
    for k in range(len(M) - 1):
        gap = raw_ids[k + 1] - raw_ids[k]
        row = [si, k, gap, 0] + [np.nan] * (len(NAMES) + 2)
        m = M[k, ..., 0] & M[k, ..., 1]
        mf = Mf[k, ..., 0]
        border = mf[0].any() or mf[-1].any() or mf[:, 0].any() or mf[:, -1].any()
        if det[k] and det[k + 1] and m.sum() >= 400 and (BORDER_OK or not border):
            r, c = np.nonzero(m)
            f = Fc[k][r, c, :2].astype(np.float64)
            x0, y0 = K[k, 0] * c + K[k, 2], K[k, 1] * r + K[k, 3]
            x1, y1 = K[k + 1, 0] * (c + f[:, 0]) + K[k + 1, 2], K[k + 1, 1] * (r + f[:, 1]) + K[k + 1, 3]
            xy, flow = np.stack([x0, y0], 1), np.stack([x1 - x0, y1 - y0], 1)
            # the ground just under the feet, from the full frame
            X, Y = Kf[k, 0] * uu + Kf[k, 2], Kf[k, 1] * vv + Kf[k, 3]
            xlo, xhi = np.percentile(x0, [2, 98])
            ylo, yhi = np.percentile(y0, [2, 98])
            w, h = xhi - xlo, yhi - ylo
            free = ~binary_dilation(mf, iterations=2)
            g = free & (X > xlo - 0.15 * w) & (X < xhi + 0.15 * w) & (Y > yhi + 0.03 * h) & (Y < yhi + 0.35 * h)
            how = 1
            if g.sum() < 25:  # the cat stands at the frame's lower edge: the ground beside its lower half
                g = free & (X > xlo - 0.3 * w) & (X < xhi + 0.3 * w) & (Y > ylo + 0.5 * h) & (Y < yhi + 0.3 * h)
                how = 2
            if g.sum() >= 25 and np.isfinite(flow).all():
                ground = np.median(Ff[k][..., :2][g].astype(np.float64), axis=0) * Kf[k, :2]
                row = [si, k, gap, how] + pair_measures(xy, flow, ground, K[k, 0] * K[k, 1]).tolist() + ground.tolist()
                nv += 1
        rows.append(row)
    meta.append({"seq": seq, "index": si, "frames": int(len(M)), "valid_pairs": nv,
                 "image_wh": [float(256 * Kf[0, 0]), float(256 * Kf[0, 1])]})
    print("%-18s %4d frames, %4d pairs measured" % (seq, len(M), nv), flush=True)
R = np.array(rows, dtype=np.float64)
np.savez(out, rows=R, cols=np.array(["seq", "k", "gap", "ground_how"] + NAMES + ["gx", "gy"]), seqs=np.array(seqs))
json.dump(meta, open(out.replace(".npz", ".json"), "w"), indent=1)
ok = R[:, 3] > 0
print("pairs: %d, measured %d (ground under the feet %d, beside %d)" % (len(R), ok.sum(), (R[:, 3] == 1).sum(), (R[:, 3] == 2).sum()))
