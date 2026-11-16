"""Behaviour demo (not a test): the trained behaviour model walks the cat around its home on its own.

Each round it picks a goal, a route to it and a body motion along it (Agent-to-Sim's own goal -> path -> body chain,
each sampled, nothing taken from the recording), moves on 5.6 s, and repeats. What the person does is set by B1_MODE:

  heldout (default): start from moments in the held-back recordings; the person stays where they stood.
  spots:   step S1 (PROTOCOL_STIMULUS.md): the person stands at each of the 12 spots in S1_SPOTS (spots.json);
           3 starting moments x B1_SEEDS seeds x B1_ROUNDS rounds each.
  replay:  step S1: the two held-back recordings, from the real cat's first 0.8 s; the person moves exactly as
           recorded, round by round, until the recording ends (needs B1_SOURCE, the released training file).
  call:    step S2: "call the cat" to each of the 12 places in S2_PLACES (places.json): the first round's goal is
           the place (given, not sampled); the route and body are the model's; later rounds are its own choice.
           The person stays where they stood. 3 starting moments x B1_SEEDS seeds x B1_ROUNDS rounds.

Run from agent2sim/gdmdm with lab4d on PYTHONPATH:
  B1_OUT=dir B1_STARTS=...-heldout-L64-S10.pkl [B1_MODE=... B1_ROUNDS=12 B1_STARTS_N=8 B1_SEEDS=2 S1_SPOTS=...
      B1_SOURCE=... B1_NAME=rollouts] python b1_rollout.py --load_logname NAME --logname_gd b128 --eval_batch_size 1 \
      --suffix latest
Writes B1_OUT/<B1_NAME>.json: per walk, the cat's position (world frame, metres, 10 fps), orientation (3x3
row-major), joint angles, and the person's position at every frame.
"""
import json
import os
import pickle
import sys
import tempfile

sys.path.insert(0, os.getcwd())
sys.path.insert(1, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch

from agent import Agent
from config import get_config
from trainer import GDMDMTrainer
from utils import get_lab4d_data

T_PAST, T_ROUND = 8, 56  # 0.8 s of past (frames 0..7), 5.6 s per round, at 10 fps


def held_back_recordings(src, hold_back=2):
    """the last two videos of the released file: their first 64-frame window (as a tiny pkl) and the person's
    world position at every frame"""
    from b1_split import KEYS, same_video
    with open(src, "rb") as f:
        d = pickle.load(f)
    n = len(d["se3"])
    starts = [0] + [i + 1 for i in range(n - 1) if not same_video(d, i)]
    videos = list(zip(starts, starts[1:] + [n]))
    held = list(range(len(videos) - hold_back, len(videos)))
    first = {k: [d[k][videos[v][0]] for v in held] for k in KEYS}
    tmp = tempfile.NamedTemporaryFile(suffix=".pkl", delete=False)
    pickle.dump(first, tmp)
    tmp.close()
    paths = []
    for v in held:
        s, e = videos[v]
        cam = np.concatenate([np.asarray(d["cam_se3"][s])] + [np.asarray(d["cam_se3"][i])[-1:] for i in range(s + 1, e)])
        paths.append(np.linalg.inv(cam)[:, :3, 3])
    return tmp.name, held, paths


@torch.no_grad()
def main():
    config = get_config()
    out_dir = os.environ["B1_OUT"]
    os.makedirs(out_dir, exist_ok=True)
    mode = os.environ.get("B1_MODE", "heldout")
    rounds = int(os.environ.get("B1_ROUNDS", 12))
    seeds = int(os.environ.get("B1_SEEDS", 2))
    assert config.eval_batch_size == 1

    rt = GDMDMTrainer.construct_test_model(config)
    model, meta = rt["model"], rt["meta"]
    plans = []  # (start index, seed, kind, extra)
    if mode == "replay":
        pkl, held, person_paths = held_back_recordings(os.environ["B1_SOURCE"])
        starts = get_lab4d_data(pkl, use_ego=not config.use_world, full_len=64, swap_cam_root=False)
        for j, v in enumerate(held):
            for seed in range(seeds):
                plans.append((j, seed, "replay", {"video": v, "path": person_paths[j],
                                                  "rounds": (len(person_paths[j]) - T_PAST) // T_ROUND}))
    else:
        starts = get_lab4d_data(os.environ["B1_STARTS"], use_ego=not config.use_world, full_len=64, swap_cam_root=False)
        n = len(starts.x0)
        if mode == "call":
            places = np.asarray(json.load(open(os.environ["S2_PLACES"]))["places"], dtype=np.float32)
            idxs = np.linspace(0, n - 1, 5).round().astype(int)[1:4].tolist()
            for k, place in enumerate(places):
                for i in idxs:
                    for seed in range(seeds):
                        plans.append((i, seed, "call", {"place": k, "pos": place, "rounds": rounds}))
        elif mode == "spots":
            spots = np.asarray(json.load(open(os.environ["S1_SPOTS"]))["spots"], dtype=np.float32)
            idxs = np.linspace(0, n - 1, 5).round().astype(int)[1:4].tolist()
            for k, spot in enumerate(spots):
                for i in idxs:
                    for seed in range(seeds):
                        plans.append((i, seed, "spot", {"spot": k, "pos": spot, "rounds": rounds}))
        else:
            idxs = np.linspace(0, n - 1, int(os.environ.get("B1_STARTS_N", 8))).round().astype(int).tolist()
            for i in idxs:
                for seed in range(seeds):
                    plans.append((i, seed, "free", {"rounds": rounds}))
    agent = Agent(model, starts, config, None, meta, agent_class=meta["agent_class"], record_grad=False)

    def r3(a):
        return np.round(np.asarray(a, dtype=np.float64), 3).tolist()

    runs = []
    for i, seed, kind, ex in plans:
        torch.manual_seed(1000 * i + seed + (7919 * ex.get("spot", 0)) + (104729 * ex.get("place", 0)))
        agent.reset([i])
        to_world = agent.data.x0_to_world[0, 0].clone()  # (1,3)
        past = (agent.data.past[0, :, 0] + to_world).cpu().numpy()
        past_joints = agent.data.past_joints[0].reshape(len(past), -1, 3).cpu().numpy()
        person = [(agent.data.cam[0, :, 0] + to_world).cpu().numpy()]  # the person during the past 0.8 s
        pos, rot, jnt = [], [], []
        for r in range(ex["rounds"]):
            now = agent.data.x0_to_world  # (1,1,1,3): where the cat is now
            if kind == "spot":
                spot = torch.as_tensor(ex["pos"], device=now.device).view(1, 1, 1, 3)
                agent.data.cam = (spot - now).expand(-1, T_PAST, -1, -1).contiguous()
            elif kind == "replay":
                t_now = T_PAST - 1 + T_ROUND * r
                seen = torch.as_tensor(ex["path"][t_now - T_PAST + 1:t_now + 1], dtype=torch.float32,
                                       device=now.device).view(1, T_PAST, 1, 3)
                agent.data.cam = seen - now
            if agent.model.global_env:
                agent.extract_local_feature()
            else:
                agent.extract_global_feature()
            if kind == "call" and r == 0:  # the called place is this round's goal
                agent.update_goal([ex["pos"].tolist()], replace_goal=True)
                agent.data.x0_goal = agent.data.x0_goal.reshape(1, 1, 1, 3)
            else:
                agent.update_goal([], replace_goal=True)
            agent.update_waypoint(replace_wp=True)
            agent.update_fullbody()
            wp = agent.reverse_wp[-1, 0, 0, :, 0] + agent.data.x0_to_world[0, 0]  # (T,3) world
            pos.append(wp.cpu().numpy())
            rot.append(agent.reverse_angles[-1, 0, 0, :, 0].reshape(-1, 9).cpu().numpy())
            jnt.append(agent.reverse_joints[-1, 0, 0].reshape(len(wp), -1, 3).cpu().numpy())  # joint angles
            if kind == "spot":
                person.append(np.repeat(ex["pos"][None], len(wp), 0))
            elif kind == "replay":
                t_now = T_PAST - 1 + T_ROUND * r
                person.append(ex["path"][t_now + 1:t_now + 1 + len(wp)])
            else:
                person.append(np.repeat(person[0][-1:], len(wp), 0))
            step = agent.reverse_wp[-1, :1, 0][:, -1:].clone()  # (1,1,1,3): where this round ends
            agent.update()
            if kind in ("free", "call"):
                agent.data.cam = agent.data.cam - step  # the person stays put
        pos, rot, jnt = np.concatenate(pos), np.concatenate(rot), np.concatenate(jnt)
        run = {"kind": kind, "start_window": i, "seed": seed, "past": r3(past), "past_joints": r3(past_joints),
               "pos": r3(pos), "rot": r3(rot), "joints": r3(jnt), "person": r3(np.concatenate(person)),
               "observer": r3(person[0][-1])}
        if kind == "spot":
            run["spot"] = ex["spot"]
        if kind == "call":
            run["place"] = ex["place"]
            run["goal"] = r3(ex["pos"])
        if kind == "replay":
            run["video"] = ex["video"]
        runs.append(run)
        print("%s %s start %d seed %d: %.1f s, moved %.2f m" % (
            kind, ex.get("spot", ex.get("video", ex.get("place", ""))), i, seed, len(pos) / 10,
            float(np.linalg.norm(np.diff(pos, axis=0), axis=1).sum())), flush=True)
    with open(os.path.join(out_dir, os.environ.get("B1_NAME", "rollouts") + ".json"), "w") as f:
        json.dump({"fps": 10, "mode": mode, "seconds_per_round": 5.6, "model": config.logname_gd, "runs": runs}, f)


if __name__ == "__main__":
    main()
