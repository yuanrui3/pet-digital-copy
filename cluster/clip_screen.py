"""Screening the close-up clips of this cat for a rebuild (work/pets/PROTOCOL_CAT.md, "Better data: screening the
close-up clips"): Lab4D's cat-85 folders 0055 to 0084. Per clip, from the archive's outlines and the judge's pair
measures (judge_real.py): length, picture size, how often the cat is whole in the frame, how large it is, how much
side-view walking it holds (the judge's windows), which way it walks, and the set's estimated view direction. Also a
picture sheet: four frames per clip. CPU only.
usage: python clip_screen.py REAL_DIR REAL_PAIRS_NPZ FRAMES_DIR OUT_DIR
  REAL_DIR: experiments_judge/real (unpacked arrays, kept_frames.json); FRAMES_DIR: .../JPEGImages/Full-Resolution"""
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import judge_score as js  # noqa: E402

real, pairs_npz, frames_dir, out = sys.argv[1:5]
os.makedirs(out, exist_ok=True)
FPS = 30.0  # assumed
base = os.path.join(real, "database", "processed")
kept = json.load(open(os.path.join(real, "kept_frames.json")))
z = np.load(pairs_npz, allow_pickle=True)
R, cols, seqs = z["rows"], list(z["cols"]), list(z["seqs"])
c = {n: i for i, n in enumerate(cols)}
meta = {m["seq"]: m for m in json.load(open(pairs_npz.replace(".npz", ".json")))}
clips = [s for s in seqs if s.startswith("cat-85-") and 55 <= int(s[-4:]) <= 84]
rows, tiles = [], []
for s in clips:
    si = seqs.index(s)
    S = R[R[:, c["seq"]] == si]
    ok = (S[:, c["ground_how"]] > 0) & np.isfinite(S[:, 4:]).all(1)
    Mf = np.load(os.path.join(base, "Annotations", "Full-Resolution", s, "full-256.npy"))[..., 0]
    border = np.array([m[0].any() or m[-1].any() or m[:, 0].any() or m[:, -1].any() for m in Mf])
    W, H = meta[s]["image_wh"] if s in meta else (np.nan, np.nan)
    mark, k, nwin = np.zeros(len(S), bool), 0, 0
    while k + js.NPAIR <= len(S):
        if ok[k:k + js.NPAIR].all():
            Wd = S[k:k + js.NPAIR]
            t = np.hypot(Wd[:, c["tx"]], Wd[:, c["ty"]])
            a = np.median(Wd[:, c["w"]] / Wd[:, c["h"]])
            net = [Wd[:, c["tx"]].sum(), Wd[:, c["ty"]].sum()]
            if a >= js.ASPECT and np.degrees(np.arctan2(abs(net[1]), abs(net[0]) + 1e-12)) <= 30 and t.sum() / Wd[:, c["gap"]].sum() >= 0.01:
                mark[k:k + js.NPAIR] = True
                nwin += 1
            k += js.HOP
        else:
            k += 1
    cam = np.load(os.path.join(base, "Cameras", "Full-Resolution", s, "01-canonical.npy"))[:, :3, :3].astype(np.float64)
    d = np.einsum("nji,j->ni", cam, np.array([0.0, 0.0, 1.0]))  # the view direction in the cat's frame (R^T z)
    az = np.degrees(np.arctan2(d[:, 0], d[:, 2]))
    el = np.degrees(np.arcsin(np.clip(d[:, 1], -1, 1)))
    walk = S[mark]
    speed = np.hypot(S[ok][:, c["tx"]], S[ok][:, c["ty"]]) / S[ok][:, c["gap"]] * FPS if ok.any() else np.array([np.nan])
    rows.append({
        "clip": s[-4:], "kept_frames": len(kept[s]), "raw_frames": int(kept[s][-1] - kept[s][0] + 1),
        "seconds": round((kept[s][-1] - kept[s][0] + 1) / FPS, 1), "picture": "%dx%d" % (round(W), round(H)),
        "whole_in_frame_share": round(float(1 - border.mean()), 2),
        "cat_size_px": int(np.nanmedian(S[:, c["size_px"]])) if ok.any() else None,
        "pairs_measured_share": round(float(ok.mean()), 2),
        "side_walk_windows": nwin, "side_walk_seconds": round(float(walk[:, c["gap"]].sum() / FPS), 1),
        "walks_towards": ("right" if np.median(walk[:, c["tx"]]) > 0 else "left") if len(walk) else "",
        "speed_sizes_per_s": round(float(np.nanmedian(speed)), 2),
        "longer_than_tall": round(float(np.nanmedian(S[ok][:, c["w"]] / S[ok][:, c["h"]])), 2) if ok.any() else None,
        "view_azimuth_deg_p10_p50_p90": [int(v) for v in np.percentile(az, [10, 50, 90])],
        "view_elevation_deg_median": int(np.median(el)),
    })
    # four frames of the clip
    names = [kept[s][int(round(q * (len(kept[s]) - 1)))] for q in (0.1, 0.37, 0.63, 0.9)]
    row = []
    for n in names:
        fp = os.path.join(frames_dir, s, "%05d.jpg" % n)
        if not os.path.exists(fp) and os.environ.get("FRAMES2"):  # a clip unpacked elsewhere
            fp = os.path.join(os.environ["FRAMES2"], s, "%05d.jpg" % n)
        im = Image.open(fp).convert("RGB")
        hgt = 150
        im = im.resize((max(1, int(im.width * hgt / im.height)), hgt))
        cell = Image.new("RGB", (270, hgt), (20, 20, 20))
        cell.paste(im.crop((max(0, (im.width - 270) // 2), 0, max(0, (im.width - 270) // 2) + min(270, im.width), hgt)), ((270 - min(270, im.width)) // 2, 0))
        row.append(np.asarray(cell))
    strip = Image.fromarray(np.concatenate(row, 1))
    ImageDraw.Draw(strip).text((6, 4), "%s  %.0f s, walking %.1f s" % (s[-4:], rows[-1]["seconds"], rows[-1]["side_walk_seconds"]), fill=(255, 255, 0))
    tiles.append(np.asarray(strip))
    print(json.dumps(rows[-1]), flush=True)
json.dump(rows, open(os.path.join(out, "clip_screen.json"), "w"), indent=1)
half = (len(tiles) + 1) // 2
Image.fromarray(np.concatenate(tiles[:half], 0)).save(os.path.join(out, "clips_1.jpg"), quality=80)
Image.fromarray(np.concatenate(tiles[half:], 0)).save(os.path.join(out, "clips_2.jpg"), quality=80)
print("clips %d; total %.0f s; side-view walking %.1f s in %d clips" % (
    len(rows), sum(r["seconds"] for r in rows), sum(r["side_walk_seconds"] for r in rows), sum(r["side_walk_seconds"] > 0 for r in rows)))
