"""Behaviour step B2 (work/pets/PROTOCOL_BEHAVIOUR.md, "B2"): better goal choice at walk time, no retraining.
"Copy + places": each of the 16 goals is either the copy's own sampled goal or a place where the cat ended up in the
training windows (B1's location-prior cells), restricted to places within reach of where the cat is now.

Run from agent2sim/gdmdm with lab4d on PYTHONPATH:
  B1_HELDOUT=...-heldout-L64-S10.pkl B2_OUT=dir python b2_goal.py --load_logname NAME --logname_gd b128 \
      --eval_batch_size 192 --eval_ntrial 12 --suffix latest
Writes B2_OUT/b2_results.json, B2_OUT/b2_examples.png and B2_OUT/places_prior.json (the cells, their counts and the
reach, for the walks).
"""
import json
import os
import sys

sys.path.insert(0, os.getcwd())
sys.path.insert(1, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch

from agent import Agent
from b1_eval import features, nearest
from config import get_config
from trainer import GDMDMTrainer
from utils import get_lab4d_data

K = 16
W_RULE = 0.5  # share of the 16 goals taken from places (fixed in the plan)
WEIGHTS = tuple(sorted({0.0, 0.25, 0.5, 0.75, 1.0, *(float(x) for x in os.environ.get("B2_EXTRA_W", "").split() if x)}))
REACH_Q = 0.99
CELL = 0.1


def scores(X, y):
    """X (n, ntrial, K, 3) guesses, y (n, 3) truth -> per-window (n,) minDE, energy score, one-goal error, and the
    trial-to-trial SD of minDE over the whole set"""
    d = (X - y[:, None, None]).norm(dim=-1)  # (n, ntrial, K)
    k = X.shape[2]
    pair = torch.cdist(X.reshape(-1, k, 3), X.reshape(-1, k, 3)).reshape(*X.shape[:3], k)
    spread = pair.sum((-1, -2)) / (k * (k - 1))  # mean distance between distinct guesses
    mde = d.min(-1).values  # (n, ntrial)
    return {"minDE": mde.mean(1), "energy": (d.mean(-1) - 0.5 * spread).mean(1), "one_goal": d.mean(-1).mean(1),
            "minDE_sd_over_trials": float(mde.mean(0).std()) if X.shape[1] > 1 else 0.0}


@torch.no_grad()
def main():
    config = get_config()
    out_dir = os.environ["B2_OUT"]
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(0)
    np.random.seed(0)
    ntrial = config.eval_ntrial
    assert config.eval_batch_size == K * ntrial
    rt = GDMDMTrainer.construct_test_model(config)
    model, meta, train = rt["model"], rt["meta"], rt["data"]
    test = get_lab4d_data(os.environ["B1_HELDOUT"], use_ego=not config.use_world, full_len=64, swap_cam_root=False)
    n = len(test.x0)
    dev = test.x0.device
    now = test.x0_to_world.reshape(n, 3)
    truth = test.x0_goal.reshape(n, 3)  # relative to the cat now, as the copy's goals are
    print("held-back windows:", n, "training windows:", len(train.x0), flush=True)

    # ---- the copy's own goals (the authors' sampling, as in B1)
    agent = Agent(model, test, config, None, meta, agent_class=meta["agent_class"], record_grad=False)
    own = []
    for s in range(0, n, 2):
        idx = list(range(s, min(s + 2, n)))
        agent.reset(idx)
        if agent.model.global_env:
            agent.extract_local_feature()
        else:
            agent.extract_global_feature()
        agent.update_goal([], replace_goal=False)
        own.append(agent.reverse_goal[-1][:, :, 0, 0])  # (bs, 192, 3)
        if s % 40 == 0:
            print("copy: %d/%d windows" % (s + len(idx), n), flush=True)
    own = torch.cat(own).view(n, ntrial, K, 3)

    # ---- places: where the cat ended up in training (B1's cells), and the reach
    ends = (train.x0_to_world[:, 0] + train.x0_goal[:, 0]).reshape(-1, 3)
    cells, counts = torch.unique(torch.floor(ends / CELL), dim=0, return_counts=True)
    centres = (cells + 0.5) * CELL
    reach = float(torch.quantile(train.x0_goal.reshape(-1, 3).norm(dim=-1), REACH_Q))

    def draw_places(pos, m, within=True):
        """m places for each position in pos (p, 3), as offsets from it; jittered inside the cell as in B1"""
        out, fallback = [], 0
        for p in pos:
            w = counts.float()
            if within:
                ok = (centres - p).norm(dim=-1) <= reach
                if ok.any():
                    w = w * ok
                else:
                    fallback += 1
            pick = torch.multinomial(w, m, replacement=True)
            out.append((cells[pick] + torch.rand(m, 3, device=dev)) * CELL - p)
        return torch.stack(out), fallback

    places, fallback = draw_places(now, ntrial * K)
    places = places.view(n, ntrial, K, 3)
    prior_all, _ = draw_places(now, ntrial * K, within=False)  # B1's location prior, drawn again
    prior_all = prior_all.view(n, ntrial, K, 3)

    # ---- Replay (B1's): the 16 most similar recorded moments
    nn_goal = nearest(features(test), features(train))  # (n, 16) indices into the training windows
    replay = train.x0_goal[nn_goal].reshape(n, 1, K, 3)

    methods = {}
    for w in WEIGHTS:
        m = int(round(K * w))
        X = torch.cat([own[:, :, :K - m], places[:, :, :m]], 2)
        methods["copy_w%.2f" % w] = scores(X, truth)
    methods["location_prior_B1"] = scores(prior_all, truth)
    methods["replay"] = scores(replay, truth)

    def mean(k, s):
        return float(methods[k][s].mean())

    mix, old, alone = "copy_w%.2f" % W_RULE, "copy_w0.00", "copy_w1.00"
    passes = (mean(mix, "minDE") < mean("location_prior_B1", "minDE") and mean(mix, "minDE") < mean("replay", "minDE")
              and mean(mix, "energy") < mean(old, "energy"))
    nothing_added = mean(alone, "minDE") < mean(mix, "minDE") and mean(alone, "energy") < mean(mix, "energy")
    table = {k: {s: (float(v[s].mean()) if s != "minDE_sd_over_trials" else v[s]) for s in v} for k, v in methods.items()}
    out = {
        "heldout_windows": n, "trials": ntrial, "goals_per_set": K, "units": "metres (3D)",
        "reach_m": reach, "reach_quantile": REACH_Q, "windows_with_no_place_in_reach": fallback,
        "rule_weight": W_RULE, "results": table,
        "passes": bool(passes),
        "readout": ("passes" if passes else "does not pass") + (
            "; the copy's own goals add nothing beyond favourite places" if nothing_added else ""),
        "share_windows_mix_minDE_below_location_prior": float(
            (methods[mix]["minDE"] < methods["location_prior_B1"]["minDE"]).float().mean()),
        "share_windows_mix_minDE_below_old_copy": float((methods[mix]["minDE"] < methods[old]["minDE"]).float().mean()),
        "per_window": {k: {s: [round(float(x), 4) for x in v[s]] for s in ("minDE", "energy")} for k, v in methods.items()},
    }
    with open(os.path.join(out_dir, "b2_results.json"), "w") as f:
        json.dump(out, f, indent=1)
    with open(os.path.join(out_dir, "places_prior.json"), "w") as f:
        json.dump({"cell_m": CELL, "cells": cells.int().tolist(), "counts": counts.tolist(), "reach_m": reach,
                   "from": "where the cat ended up (goal) in every training window, 10 cm cells"}, f)

    print("reach %.2f m (%.0fth percentile of training goal distances); windows with no place in reach: %d" % (
        reach, 100 * REACH_Q, fallback))
    print("%-20s %8s %8s %8s %8s" % ("method", "minDE", "sd", "energy", "one-goal"))
    for k, v in table.items():
        print("%-20s %8.3f %8.4f %8.3f %8.3f" % (k, v["minDE"], v["minDE_sd_over_trials"], v["energy"], v["one_goal"]))
    print("B2:", out["readout"], flush=True)

    # ---- six examples from above: past, true goal, old copy's 16 goals, the mix's 16
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 6, figsize=(24, 4.4))
    mm = int(round(K * W_RULE))
    for ax, i in zip(axes, np.linspace(0, n - 1, 6).round().astype(int)):
        p = test.past[i, :, 0].cpu()
        ax.plot(p[:, 0], p[:, 2], "k-", lw=2, label="past 0.8 s")
        t = truth[i].cpu()
        ax.plot(t[0], t[2], "g*", ms=14, label="where it went")
        o = own[i, 0].cpu()
        ax.plot(o[:, 0], o[:, 2], "r.", ms=9, label="old copy: 16 goals")
        x = torch.cat([own[i, 0, :K - mm], places[i, 0, :mm]]).cpu()
        ax.plot(x[K - mm:, 0], x[K - mm:, 2], "b^", ms=6, label="mix: places")
        ax.plot(x[:K - mm, 0], x[:K - mm, 2], "ro", ms=7, mfc="none", label="mix: copy's own")
        ax.set_aspect("equal")
        ax.set_title("window %d" % i)
    axes[0].legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "b2_examples.png"), dpi=90)


if __name__ == "__main__":
    main()
