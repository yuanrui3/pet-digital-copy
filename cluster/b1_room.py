"""Step S2 (work/pets/PROTOCOL_STIMULUS.md, "S2"), checks: does the copy's choice of goal depend on the room, and on
whether the person is walking toward it or away? Also picks the 12 places for the "call the cat" demo (k-means of
where the real cat has been).

Run from agent2sim/gdmdm with lab4d on PYTHONPATH:
  B1_HELDOUT=...-heldout-L64-S10.pkl S2_OUT=dir python b1_room.py --load_logname NAME --logname_gd b128 \
      --eval_batch_size 16 --suffix latest
Writes S2_OUT/s2_check.json and S2_OUT/places.json.
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
SPEED = 0.5  # m/s for the walking person (0.4 m over the 0.8 s the model sees)
NEAR_VISITED = 0.25  # m: a goal counts as "where the cat goes" if the real cat's root was this close (3D)


def kmeans(x, k, iters=60, seed=0):
    rng = np.random.default_rng(seed)
    c = x[rng.choice(len(x), k, replace=False)]
    for _ in range(iters):
        lab = ((x[:, None] - c[None]) ** 2).sum(-1).argmin(1)
        c = np.stack([x[lab == j].mean(0) if (lab == j).any() else c[j] for j in range(k)])
    return c, np.bincount(lab, minlength=k)


def readout(ratio):
    return "responds" if ratio >= 2 else ("barely" if ratio >= 1 else "ignores")


@torch.no_grad()
def main():
    config = get_config()
    assert config.eval_batch_size == K
    out_dir = os.environ["S2_OUT"]
    os.makedirs(out_dir, exist_ok=True)
    rt = GDMDMTrainer.construct_test_model(config)
    model, meta, train = rt["model"], rt["meta"], rt["data"]
    test = get_lab4d_data(os.environ["B1_HELDOUT"], use_ego=not config.use_world, full_len=64, swap_cam_root=False)
    n = len(test.x0)
    agent = Agent(model, test, config, None, meta, agent_class=meta["agent_class"], record_grad=False)
    gm = model.goal_model
    gm_forward = gm.forward

    def goals(idx, seed, cam=None, no_room=False):
        torch.manual_seed(seed)
        agent.reset(idx)
        if cam is not None:
            agent.data.cam = cam
        if agent.model.global_env:
            agent.extract_local_feature()
        else:
            agent.extract_global_feature()
        if no_room:  # the scene embedding set to zero, as the model's own training dropout does
            gm.forward = lambda *a, **k: gm_forward(*a, **dict(k, drop_feat=True))
        try:
            agent.update_goal([], replace_goal=False)
        finally:
            gm.forward = gm_forward
        g = agent.reverse_goal[-1][:, :, 0, 0]  # (bs, K, 3), relative to where the cat is now
        return (g + agent.data.x0_to_world[:, 0]).cpu().numpy()

    h = [0, 2]  # horizontal axes (y is height, pointing down)
    names = ("A1", "A2", "N", "App", "Ret", "Still")
    rows = {k: [] for k in names}
    cat_now, person_now, real_goal = [], [], []
    for s in range(0, n, 4):
        idx = list(range(s, min(s + 4, n)))
        agent.reset(idx)
        cam = agent.data.cam.clone()  # (bs, 8, 1, 3) relative to the cat now; frame 7 is "now"
        p = cam[:, -1:, :, :].clone()  # (bs,1,1,3)
        d = p.clone()
        d[..., 1] = 0
        d = d / d.norm(dim=-1, keepdim=True).clamp(min=1e-6)  # horizontal unit vector, cat -> person
        back = torch.arange(7, -1, -1, device=cam.device, dtype=cam.dtype).view(1, 8, 1, 1) * 0.1 * SPEED
        approach = p + d * back  # earlier frames farther away: walking toward the cat
        retreat = p - d * back  # earlier frames closer: walking away
        still = p.expand(-1, 8, -1, -1).contiguous()
        to_world = agent.data.x0_to_world[:, 0, 0]
        cat_now.append(to_world.cpu().numpy())
        person_now.append((p[:, 0, 0] + to_world).cpu().numpy())
        real_goal.append((agent.data.x0_goal[:, 0, 0] + to_world).cpu().numpy())
        rows["A1"].append(goals(idx, 1))
        rows["A2"].append(goals(idx, 2))
        rows["N"].append(goals(idx, 1, no_room=True))
        rows["App"].append(goals(idx, 1, cam=approach))
        rows["Ret"].append(goals(idx, 1, cam=retreat))
        rows["Still"].append(goals(idx, 1, cam=still))
        if s % 40 == 0:
            print("%d/%d moments" % (s + len(idx), n), flush=True)
    G = {k: np.concatenate(v) for k, v in rows.items()}  # (n, K, 3) world
    C0, P, RG = (np.concatenate(x) for x in (cat_now, person_now, real_goal))
    centre = {k: v.mean(1)[:, h] for k, v in G.items()}
    noise = np.linalg.norm(centre["A2"] - centre["A1"], axis=1)

    # Part A: the room. Where the real cat has been: every training and held-back moment's position
    tr_cat = train.x0_to_world[:, 0, 0].cpu().numpy()
    visited = torch.tensor(np.concatenate([tr_cat, C0]), device="cuda")

    def share_visited(pts):  # pts (..., 3) -> fraction within NEAR_VISITED of a visited position
        q = torch.tensor(pts.reshape(-1, 3), device="cuda")
        dmin = torch.cat([torch.cdist(q[i:i + 4096], visited).min(1).values for i in range(0, len(q), 4096)])
        return float((dmin < NEAR_VISITED).float().mean())

    shift_n = np.linalg.norm(centre["N"] - centre["A1"], axis=1)
    rn = float(np.median(shift_n) / np.median(noise))
    vis_a1, vis_n, vis_real = share_visited(G["A1"]), share_visited(G["N"]), share_visited(RG)
    drop = vis_a1 - vis_n
    place = ("keeps goals where the cat goes" if drop >= 0.10 else ("slightly" if drop >= 0.03 else "no effect on placement"))

    # Part B: the person walking toward vs away, on moments with the person 0.8 to 3 m away
    dist = np.linalg.norm(P[:, h] - C0[:, h], axis=1)
    sel = (dist >= 0.8) & (dist <= 3.0)
    u = (P[:, h] - C0[:, h]) / dist[:, None]
    diff = centre["App"] - centre["Ret"]
    shift_ar = np.linalg.norm(diff, axis=1)
    toward = (diff * u).sum(1)  # > 0: when approached, the copy heads more toward the person
    rar = float(np.median(shift_ar[sel]) / np.median(noise[sel]))
    shift_still = np.linalg.norm(centre["Still"] - centre["A1"], axis=1)

    out = {
        "moments": n, "goals_per_condition": K, "units": "metres, horizontal unless said",
        "median_noise_A2_vs_A1": float(np.median(noise)),
        "room": {
            "median_shift_no_room": float(np.median(shift_n)), "ratio": rn, "readout_shift": readout(rn),
            "share_goals_near_visited_with_room": vis_a1, "share_goals_near_visited_no_room": vis_n,
            "share_real_goals_near_visited": vis_real, "drop": drop, "readout_placement": place,
            "near_visited_threshold_m_3d": NEAR_VISITED,
            "median_goal_height_with_room_m": float(np.median(-G["A1"][..., 1])),
            "median_goal_height_no_room_m": float(np.median(-G["N"][..., 1])),
        },
        "walking_person": {
            "moments_used": int(sel.sum()), "speed_m_s": SPEED,
            "median_noise_on_these": float(np.median(noise[sel])),
            "median_shift_approach_vs_retreat": float(np.median(shift_ar[sel])), "ratio": rar,
            "readout": readout(rar),
            "median_toward_person_when_approached_m": float(np.median(toward[sel])),
            "share_moments_toward_when_approached": float((toward[sel] > 0).mean()),
            "median_shift_still_vs_real_motion": float(np.median(shift_still[sel])),
        },
        "per_moment": {"noise": np.round(noise, 3).tolist(), "shift_no_room": np.round(shift_n, 3).tolist(),
                       "shift_app_ret": np.round(shift_ar, 3).tolist(), "toward": np.round(toward, 3).tolist(),
                       "person_dist": np.round(dist, 3).tolist()},
    }
    with open(os.path.join(out_dir, "s2_check.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "per_moment"}, indent=1), flush=True)

    # 12 places the real cat goes, for "call the cat" (3D, so a sofa top and the floor below it stay apart)
    places, counts = kmeans(np.concatenate([tr_cat, C0]), 12)
    with open(os.path.join(out_dir, "places.json"), "w") as f:
        json.dump({"places": np.round(places, 3).tolist(), "moments_per_place": counts.tolist(),
                   "from": "k-means (k=12, seed 0) of the real cat's position at every training and held-back moment"}, f)
    print("places:", np.round(places, 2).tolist(), counts.tolist(), flush=True)


if __name__ == "__main__":
    main()
