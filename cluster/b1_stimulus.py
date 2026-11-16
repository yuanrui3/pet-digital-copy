"""Step S1 (work/pets/PROTOCOL_STIMULUS.md), Part 2: does the copy's choice of goal depend on where the person is?
Also picks the 12 person spots for the demo (k-means of where people really stood while filming).

Run from agent2sim/gdmdm with lab4d on PYTHONPATH:
  B1_HELDOUT=...-heldout-L64-S10.pkl S1_OUT=dir python b1_stimulus.py --load_logname NAME --logname_gd b128 \
      --eval_batch_size 16 --suffix latest
Writes S1_OUT/s1_check.json and S1_OUT/spots.json.
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

K = 16


def kmeans(x, k, iters=60, seed=0):
    rng = np.random.default_rng(seed)
    c = x[rng.choice(len(x), k, replace=False)]
    for _ in range(iters):
        lab = ((x[:, None] - c[None]) ** 2).sum(-1).argmin(1)
        c = np.stack([x[lab == j].mean(0) if (lab == j).any() else c[j] for j in range(k)])
    return c, np.bincount(lab, minlength=k)


@torch.no_grad()
def main():
    config = get_config()
    assert config.eval_batch_size == K
    out_dir = os.environ["S1_OUT"]
    os.makedirs(out_dir, exist_ok=True)
    rt = GDMDMTrainer.construct_test_model(config)
    model, meta, train = rt["model"], rt["meta"], rt["data"]
    test = get_lab4d_data(os.environ["B1_HELDOUT"], use_ego=not config.use_world, full_len=64, swap_cam_root=False)
    n = len(test.x0)
    agent = Agent(model, test, config, None, meta, agent_class=meta["agent_class"], record_grad=False)
    drop_default = agent.drop_cam

    def goals(idx, seed, cam=None, drop=False):
        torch.manual_seed(seed)
        agent.reset(idx)
        if cam is not None:
            agent.data.cam = cam
        agent.drop_cam = drop
        if agent.model.global_env:
            agent.extract_local_feature()
        else:
            agent.extract_global_feature()
        agent.update_goal([], replace_goal=False)
        agent.drop_cam = drop_default
        g = agent.reverse_goal[-1][:, :, 0, 0]  # (bs, K, 3), relative to where the cat is now
        return (g + agent.data.x0_to_world[:, 0]).cpu().numpy()

    rows = {k: [] for k in ("A1", "A2", "B", "C")}
    person_now, person_mirror, cat_now, real_goal = [], [], [], []
    for s in range(0, n, 4):
        idx = list(range(s, min(s + 4, n)))
        agent.reset(idx)
        cam = agent.data.cam.clone()  # (bs, 8, 1, 3), relative to the cat now
        mirror = cam.clone()
        mirror[..., [0, 2]] *= -1  # same distance, opposite side (horizontal)
        to_world = agent.data.x0_to_world[:, 0, 0]  # (bs, 3)
        person_now.append((cam[:, -1, 0] + to_world).cpu().numpy())
        person_mirror.append((mirror[:, -1, 0] + to_world).cpu().numpy())
        cat_now.append(to_world.cpu().numpy())
        real_goal.append((agent.data.x0_goal[:, 0, 0] + to_world).cpu().numpy())
        rows["A1"].append(goals(idx, 1))
        rows["A2"].append(goals(idx, 2))
        rows["B"].append(goals(idx, 1, cam=mirror))
        rows["C"].append(goals(idx, 1, drop=True))
        if s % 40 == 0:
            print("%d/%d moments" % (s + len(idx), n), flush=True)
    G = {k: np.concatenate(v) for k, v in rows.items()}  # (n, K, 3)
    P, PM, C0, RG = (np.concatenate(x) for x in (person_now, person_mirror, cat_now, real_goal))
    h = [0, 2]  # horizontal axes (y is height, pointing down)
    centre = {k: v.mean(1)[:, h] for k, v in G.items()}
    noise = np.linalg.norm(centre["A2"] - centre["A1"], axis=1)
    shift_b = np.linalg.norm(centre["B"] - centre["A1"], axis=1)
    shift_c = np.linalg.norm(centre["C"] - centre["A1"], axis=1)
    u = PM[:, h] - P[:, h]
    u /= np.linalg.norm(u, axis=1, keepdims=True) + 1e-9
    toward = ((centre["B"] - centre["A1"]) * u).sum(1)

    def nearer(goal_xz, start_xz, person_xz):
        return np.linalg.norm(goal_xz - person_xz, axis=-1) < np.linalg.norm(start_xz - person_xz, axis=-1)

    copy_nearer = nearer(G["A1"][..., h], C0[:, None, h], P[:, None, h]).mean()
    real_nearer = nearer(RG[:, h], C0[:, h], P[:, h]).mean()
    tr_person = (train.cam[:, -1, 0] + train.x0_to_world[:, 0, 0]).cpu().numpy()
    tr_cat = train.x0_to_world[:, 0, 0].cpu().numpy()
    tr_goal = (train.x0_goal[:, 0, 0] + train.x0_to_world[:, 0, 0]).cpu().numpy()
    train_nearer = nearer(tr_goal[:, h], tr_cat[:, h], tr_person[:, h]).mean()

    def readout(ratio):
        return "responds" if ratio >= 2 else ("barely" if ratio >= 1 else "ignores")

    rb, rc = float(np.median(shift_b) / np.median(noise)), float(np.median(shift_c) / np.median(noise))
    out = {
        "moments": n, "goals_per_condition": K, "units": "metres, horizontal",
        "median_noise_A2_vs_A1": float(np.median(noise)),
        "median_shift_person_mirrored_B": float(np.median(shift_b)),
        "median_shift_no_person_C": float(np.median(shift_c)),
        "ratio_B": rb, "readout_B": readout(rb), "ratio_C": rc, "readout_C": readout(rc),
        "B_shift_toward_new_side_median_m": float(np.median(toward)),
        "B_share_of_moments_shift_toward_new_side": float((toward > 0).mean()),
        "copy_share_of_goals_nearer_person_than_start": float(copy_nearer),
        "real_cat_share_heldout_nearer_person": float(real_nearer),
        "real_cat_share_training_nearer_person": float(train_nearer),
        "per_moment": {"noise": np.round(noise, 3).tolist(), "shift_B": np.round(shift_b, 3).tolist(),
                       "shift_C": np.round(shift_c, 3).tolist(), "toward_B": np.round(toward, 3).tolist()},
    }
    with open(os.path.join(out_dir, "s1_check.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "per_moment"}, indent=1), flush=True)

    # the 12 spots where people really stood (training and held-back moments together)
    allp = np.concatenate([tr_person, P])
    spots, counts = kmeans(allp, 12)
    with open(os.path.join(out_dir, "spots.json"), "w") as f:
        json.dump({"spots": np.round(spots, 3).tolist(), "moments_per_spot": counts.tolist(),
                   "from": "k-means (k=12, seed 0) of the person's position at every training and held-back moment"}, f)
    print("spots:", np.round(spots, 2).tolist(), counts.tolist(), flush=True)


if __name__ == "__main__":
    main()
