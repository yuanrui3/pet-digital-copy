"""Applies the pass rule of work/pets/PROTOCOL.md to the fetched results and writes out/B/summary.json.

Pass rule (arm B): a dog passes if, averaged over its 50 held-back frames, the reconstruction beats the
nearest-used-frame baseline on both LPIPS (lower) and silhouette IoU (higher); the test passes if at least
2 of the 3 dogs pass. Secondary, reported but not deciding: PSNR inside the real outline against the
baseline, and the share of held-back frames where the reconstruction beats the baseline on LPIPS.

usage: python summarize.py [ARM]
"""
import json
import os
import sys

DOGS = ["1030_23106_17099", "1050_43243_38827", "1022_15133_10069"]


def main(arm="B"):
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", arm)
    rows, missing = [], []
    for seq in DOGS:
        path = os.path.join(root, seq, "heldout.json")
        if not os.path.isfile(path):
            missing.append(seq)
            continue
        with open(path) as f:
            r = json.load(f)
        log = os.path.join(root, seq, "run.log")
        early = None
        if os.path.isfile(log):
            with open(log, errors="ignore") as f:
                early = "Early stop" in f.read()
        pf = r["per_frame"]
        n = len(pf["frame"])
        lpips_wins = sum(a < b for a, b in zip(pf["avatar_lpips"], pf["baseline_lpips"]))
        rows.append({
            "dog": seq,
            "passes": r["avatar"]["lpips"] < r["baseline"]["lpips"] and r["avatar"]["iou"] > r["baseline"]["iou"],
            "lpips": [r["avatar"]["lpips"], r["baseline"]["lpips"]],
            "iou": [r["avatar"]["iou"], r["baseline"]["iou"]],
            "psnr_mask": [r["avatar"]["psnr_mask"], r["baseline"]["psnr_mask"]],
            "lpips_frames_won": f"{lpips_wins}/{n}",
            "turn_deg_median": round(r["turn_deg_median"], 1),
            "turn_deg_max": round(r["turn_deg_max"], 1),
            "early_stop_fired": early,
        })
    out = {"arm": arm, "dogs": rows, "missing": missing}
    if not missing:
        out["dogs_passing"] = sum(r["passes"] for r in rows)
        out["test_passes"] = out["dogs_passing"] >= 2
    with open(os.path.join(root, "summary.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("dog                lpips (copy / photo)   iou (copy / photo)   psnr-in-mask (copy / photo)   lpips won   turn   pass")
    for r in rows:
        print(f"{r['dog']}   {r['lpips'][0]:.3f} / {r['lpips'][1]:.3f}        {r['iou'][0]:.3f} / {r['iou'][1]:.3f}"
              f"        {r['psnr_mask'][0]:5.2f} / {r['psnr_mask'][1]:5.2f}              {r['lpips_frames_won']:>6}   "
              f"{r['turn_deg_median']:4.0f}°   {'yes' if r['passes'] else 'no'}{'  (early stop)' if r['early_stop_fired'] else ''}")
    if missing:
        print("not finished:", ", ".join(missing))
    else:
        print(f"dogs passing: {out['dogs_passing']} of 3 -> test {'PASSES' if out['test_passes'] else 'FAILS'}")


if __name__ == "__main__":
    main(*sys.argv[1:])
