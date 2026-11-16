"""Behaviour test B1 scoring (work/pets/PROTOCOL_BEHAVIOUR.md). Run from agent2sim/gdmdm with lab4d on PYTHONPATH:
  B1_HELDOUT=path/to/...-heldout-L64-S10.pkl B1_OUT=dir python b1_eval.py --load_logname NAME --logname_gd b128 \
      --eval_batch_size 192 --eval_ntrial 12 --suffix latest
The copy: the authors' own evaluation calls (demo.py / visualizer.run_eval: goals sampled freely, paths given the true
goal, body given the true path; 192 samples = 16 x 12 trials), looped over every held-back window.
Baselines, scored with the same min-of-16 rule: Replay (decides the pass rule), Location prior (goal only), Stay put.
Writes B1_OUT/b1_results.json and B1_OUT/b1_examples.png.
"""
import json
import os
import sys

sys.path.insert(0, os.getcwd())
import numpy as np
import torch

from agent import Agent
from config import get_config
from trainer import GDMDMTrainer
from utils import get_lab4d_data

K = 16  # samples per prediction (the paper's)


def per_window_min(pred, gt, ntrial, rot=False):
    """pred (bs, ntrial*K, T, J, 3[,3]), gt (bs, T, J, 3[,3]) -> (bs, ntrial): best-of-K mean error per trial,
    exactly as gdmdm/eval.py's eval_ADE before it averages over trials"""
    from lab4d.utils.geom_utils import rot_angle

    bs = pred.shape[0]
    pred = pred.reshape(bs, ntrial, -1, *pred.shape[2:])
    if rot:
        diff = rot_angle(pred @ gt[:, None, None].transpose(-1, -2))
    else:
        diff = (pred - gt[:, None, None]).norm(2, -1)
    return diff.mean(3).mean(3).min(2).values


def features(d, i=None):
    """what Replay matches on: past path and observer (relative to the cat now), room position, heading now"""
    sl = slice(None) if i is None else i
    n = d.past[sl].shape[0]
    return torch.cat([d.past[sl].reshape(n, -1), d.cam[sl].reshape(n, -1), d.x0_to_world[sl].reshape(n, -1),
                      d.past_angles[sl][:, -1].reshape(n, -1)], -1)


def nearest(query, bank, k=K, chunk=64):
    out = []
    for s in range(0, len(query), chunk):
        out.append(torch.cdist(query[s:s + chunk], bank).topk(k, largest=False).indices)
    return torch.cat(out)


@torch.no_grad()
def main():
    config = get_config()
    held_path, out_dir = os.environ["B1_HELDOUT"], os.environ["B1_OUT"]
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(0)
    np.random.seed(0)
    ntrial = config.eval_ntrial
    assert config.eval_batch_size == K * ntrial

    rt = GDMDMTrainer.construct_test_model(config)  # model + the training windows
    model, meta, train = rt["model"], rt["meta"], rt["data"]
    test = get_lab4d_data(held_path, use_ego=not config.use_world, full_len=64, swap_cam_root=False)
    n = len(test.x0)
    print("held-back windows:", n, "training windows:", len(train.x0))

    # ---- the copy, through the authors' own calls
    agent = Agent(model, test, config, None, meta, agent_class=meta["agent_class"], record_grad=False)
    res = {m: {} for m in ("copy", "replay", "location_prior", "stay_put")}
    acc = {k: [] for k in ("goal", "path", "orient", "joints")}
    goal_samples = []
    for s in range(0, n, 2):
        idx = list(range(s, min(s + 2, n)))
        agent.reset(idx)
        if agent.model.global_env:
            agent.extract_local_feature()
        else:
            agent.extract_global_feature()
        agent.update_goal([], replace_goal=False)
        agent.update_waypoint(replace_wp=False)
        agent.update_fullbody()
        g = agent.reverse_goal[-1]
        acc["goal"].append(per_window_min(g, agent.data.x0_goal, ntrial))
        acc["path"].append(per_window_min(agent.reverse_wp_guide[-1], agent.data.x0, ntrial))
        acc["orient"].append(per_window_min(agent.reverse_angles[-1], agent.data.x0_angles, ntrial, rot=True))
        acc["joints"].append(per_window_min(agent.reverse_joints[-1], agent.data.x0_joints, ntrial))
        goal_samples.append(g[:, :K, 0, 0].cpu())
        if s % 40 == 0:
            print("copy: %d/%d windows" % (s + len(idx), n), flush=True)
    copy_pw = {k: torch.cat(v) for k, v in acc.items()}  # (n, ntrial)
    goal_samples = torch.cat(goal_samples)

    # ---- Replay: the 16 most similar recorded moments, and what the cat did next in each
    ftr, fte = features(train), features(test)
    nn_goal = nearest(fte, ftr)
    nn_path = nearest(torch.cat([fte, test.x0_goal.reshape(n, -1)], -1),
                      torch.cat([ftr, train.x0_goal.reshape(len(ftr), -1)], -1))
    nn_body = nearest(torch.cat([fte, test.x0.reshape(n, -1)], -1),
                      torch.cat([ftr, train.x0.reshape(len(ftr), -1)], -1))
    replay_pw = {
        "goal": per_window_min(train.x0_goal[nn_goal], test.x0_goal, 1),
        "path": per_window_min(train.x0[nn_path], test.x0, 1),
        "orient": per_window_min(train.x0_angles[nn_body], test.x0_angles, 1, rot=True),
        "joints": per_window_min(train.x0_joints[nn_body], test.x0_joints, 1),
    }

    # ---- Location prior: goals drawn from where the cat ended up in training (10 cm cells), as seen from now
    ends = (train.x0_to_world[:, 0] + train.x0_goal[:, 0]).reshape(-1, 3)
    cells, counts = torch.unique(torch.floor(ends / 0.1), dim=0, return_counts=True)
    draw = torch.multinomial(counts.float(), n * ntrial * K, replacement=True)
    pts = (cells[draw] + torch.rand(len(draw), 3, device=cells.device)) * 0.1
    pts = pts.view(n, ntrial * K, 1, 1, 3) - test.x0_to_world.view(n, 1, 1, 1, 3)
    loc_pw = {"goal": per_window_min(pts, test.x0_goal, ntrial)}

    # ---- Stay put: no movement, body as it is now
    T = test.x0.shape[1]
    stay_pw = {
        "goal": per_window_min(torch.zeros_like(test.x0_goal)[:, None], test.x0_goal, 1),
        "path": per_window_min(torch.zeros_like(test.x0)[:, None], test.x0, 1),
        "orient": per_window_min(test.past_angles[:, -1:].expand(-1, T, -1, -1, -1)[:, None], test.x0_angles, 1,
                                 rot=True),
        "joints": per_window_min(test.past_joints[:, -1:].expand(-1, T, -1, -1)[:, None], test.x0_joints, 1),
    }

    def summary(pw):
        return {k: {"mean": float(v.mean()), "std_over_trials": float(v.mean(0).std()) if v.shape[1] > 1 else 0.0,
                    "per_window": [round(float(x), 4) for x in v.mean(1)]} for k, v in pw.items()}

    res = {"copy": summary(copy_pw), "replay": summary(replay_pw), "location_prior": summary(loc_pw),
           "stay_put": summary(stay_pw)}
    beats = {k: res["copy"][k]["mean"] < res["replay"][k]["mean"] for k in ("goal", "path", "orient", "joints")}
    out = {
        "heldout_windows": n,
        "training_windows": len(train.x0),
        "samples_per_prediction": K,
        "trials": ntrial,
        "units": {"goal": "m (minDE)", "path": "m (minADE)", "orient": "rad (minADE)", "joints": "rad (minADE)"},
        "results": res,
        "copy_beats_replay": beats,
        "passes": beats["goal"] and beats["path"],
        "share_of_windows_copy_goal_better_than_replay": float(
            (copy_pw["goal"].mean(1) < replay_pw["goal"].mean(1)).float().mean()),
        "paper_table4_ats_for_rough_comparison": {"goal": 0.448, "path": 0.234, "orient": 0.550, "joints": 0.237},
    }
    with open(os.path.join(out_dir, "b1_results.json"), "w") as f:
        json.dump(out, f, indent=1)

    print("%-15s %8s %8s %8s %8s" % ("method", "goal m", "path m", "orient", "joints"))
    for m in res:
        print("%-15s " % m + " ".join("%8.3f" % res[m][k]["mean"] if k in res[m] else "%8s" % "-"
                                      for k in ("goal", "path", "orient", "joints")))
    print("copy beats replay:", beats, "-> PASSES" if out["passes"] else "-> does not pass")

    # ---- six examples, seen from above: past, true future, the copy's 16 goals, replay's 16 goals
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 6, figsize=(24, 4.4))
    for ax, i in zip(axes, np.linspace(0, n - 1, 6).round().astype(int)):
        p, f = test.past[i, :, 0].cpu(), test.x0[i, :, 0].cpu()
        ax.plot(p[:, 0], p[:, 2], "k-", lw=2, label="past 0.8 s")
        ax.plot(f[:, 0], f[:, 2], "g-", lw=2, label="what it did (5.6 s)")
        ax.plot(f[-1, 0], f[-1, 2], "g*", ms=14)
        cg = goal_samples[i]
        ax.plot(cg[:, 0], cg[:, 2], "r.", ms=9, label="copy: 16 goals")
        rg = train.x0_goal[nn_goal[i], 0, 0].cpu()
        ax.plot(rg[:, 0], rg[:, 2], "bx", ms=7, label="replay: 16 goals")
        c = test.cam[i, -1, 0].cpu()
        ax.plot(c[0], c[2], "ms", ms=8, label="observer")
        ax.set_aspect("equal")
        ax.set_title("window %d" % i)
    axes[0].legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "b1_examples.png"), dpi=90)


if __name__ == "__main__":
    main()
