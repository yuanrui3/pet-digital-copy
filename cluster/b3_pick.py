"""Step B3 (work/pets/PROTOCOL_BEHAVIOUR.md, "B3"): pick the walks' place chance w so that walks started from
training moments move as much as the real cat does in the 17 training videos (mean horizontal movement over 5.6 s,
stretches starting every 1 s). Standard library only.
usage: python b3_pick.py OUT_DIR W1 W2 ...   (reads OUT_DIR/tune_w<W>.json, writes OUT_DIR/b3_pick.json, prints w)
"""
import json
import math
import os
import sys

TARGET = 0.541  # m, the 17 training videos (800 stretches); fixed in the plan
REAL_SHARE = {"over_0.5m": 0.350, "over_1.5m": 0.124}
TIE = 0.05 * TARGET
STEP, EVERY = 56, 10  # 5.6 s at 10 fps, a stretch starting every 1 s


def moves(runs):
    d = []
    for r in runs:
        p = r["pos"]
        d += [math.hypot(p[t + STEP][0] - p[t][0], p[t + STEP][2] - p[t][2]) for t in range(0, len(p) - STEP, EVERY)]
    return d


def main():
    out, ws = sys.argv[1], sys.argv[2:]
    rows = []
    for w in ws:
        runs = json.load(open(os.path.join(out, "tune_w%s.json" % w)))["runs"]
        d = moves(runs)
        src = [s for r in runs for s in r.get("goal_src", [])]
        rows.append({"w": float(w), "w_text": w, "walks": len(runs), "stretches": len(d), "mean_m": sum(d) / len(d),
                     "over_0.5m": sum(x > 0.5 for x in d) / len(d), "over_1.5m": sum(x > 1.5 for x in d) / len(d),
                     "rounds_to_a_place": src.count("place"), "rounds": len(src)})
    best = min(abs(r["mean_m"] - TARGET) for r in rows)
    pick = min((r for r in rows if abs(r["mean_m"] - TARGET) <= best + TIE), key=lambda r: r["w"])
    json.dump({"target_mean_m": TARGET, "real_share": REAL_SHARE, "tie_m": TIE, "rows": rows, "picked_w": pick["w"],
               "picked_w_text": pick["w_text"]}, open(os.path.join(out, "b3_pick.json"), "w"), indent=1)
    for r in rows:
        print("w %-5s mean %.3f m  over 0.5 m %4.1f%%  over 1.5 m %4.1f%%  place rounds %d/%d%s" % (
            r["w_text"], r["mean_m"], 100 * r["over_0.5m"], 100 * r["over_1.5m"], r["rounds_to_a_place"], r["rounds"],
            "   <- picked" if r is pick else ""), file=sys.stderr)
    print("real (training videos): mean %.3f m, over 0.5 m %.1f%%, over 1.5 m %.1f%%" % (
        TARGET, 100 * REAL_SHARE["over_0.5m"], 100 * REAL_SHARE["over_1.5m"]), file=sys.stderr)
    print(pick["w_text"])


if __name__ == "__main__":
    main()
