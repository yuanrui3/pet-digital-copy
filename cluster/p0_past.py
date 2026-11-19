"""Phase 0, P0-2 (work/pets/PROTOCOL_CAT.md, "Restart, Phase 0"): does the behaviour model's body stage use the past
facing? For 20 windows of the held-out recordings x 3 seeds: the goal and path are sampled once (as in b1_rollout.py,
a "free" walk), then the body stage is run three times with the same noise: as is; with the 8 past headings turned 90
degrees about the vertical (positions and joints unchanged); and with the past dropped (the model's own drop_past). The
first generated heading is compared. GPU (minutes).
Run from agent2sim/gdmdm with lab4d on PYTHONPATH, like b1_rollout.py:
  P0_OUT=dir B1_STARTS=...-heldout-L64-S10.pkl python p0_past.py --load_logname NAME --logname_gd b128 \
      --eval_batch_size 1 --suffix latest"""
import json
import math
import os
import sys

sys.path.insert(0, os.getcwd())
sys.path.insert(1, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch

from agent import Agent
from config import get_config
from trainer import GDMDMTrainer
from utils import get_lab4d_data


def yaw_of(R):
    """the facing's angle on the floor (the cat faces -z in its frame; the room's y points down), degrees"""
    f = -R[:, 2]
    return math.degrees(math.atan2(f[0], f[2]))


def wrap(a):
    return (a + 180.0) % 360.0 - 180.0


@torch.no_grad()
def main():
    config = get_config()
    out = os.environ["P0_OUT"]
    os.makedirs(out, exist_ok=True)
    assert config.eval_batch_size == 1
    rt = GDMDMTrainer.construct_test_model(config)
    model, meta = rt["model"], rt["meta"]
    starts = get_lab4d_data(os.environ["B1_STARTS"], use_ego=not config.use_world, full_len=64, swap_cam_root=False)
    n = len(starts.x0)
    idxs = np.linspace(0, n - 1, 20).round().astype(int).tolist()
    agent = Agent(model, starts, config, None, meta, agent_class=meta["agent_class"], record_grad=False)
    drop0 = agent.drop_past
    c, s = math.cos(math.radians(90)), math.sin(math.radians(90))
    rows = []
    for i in idxs:
        for seed in range(3):
            torch.manual_seed(1000 * i + seed)
            agent.reset([i])
            if agent.model.global_env:
                agent.extract_local_feature()
            else:
                agent.extract_global_feature()
            agent.update_goal([], replace_goal=True)
            agent.update_waypoint(replace_wp=True)
            base = agent.data.past_angles.clone()
            dev = base.device
            Ry = torch.tensor([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=base.dtype, device=dev)
            last = base.reshape(-1, 3, 3)[-1].cpu().numpy()
            first = {}
            for cond in ("as is", "past turned 90", "past dropped"):
                agent.data.past_angles = base.clone()
                agent.drop_past = drop0
                if cond == "past turned 90":
                    agent.data.past_angles = (Ry @ base.reshape(-1, 3, 3)).reshape(base.shape)
                if cond == "past dropped":
                    agent.drop_past = True
                torch.manual_seed(77 + seed)
                agent.update_fullbody()
                first[cond] = agent.reverse_angles[-1, 0, 0, 0, 0].reshape(3, 3).cpu().numpy()
            agent.data.past_angles = base
            agent.drop_past = drop0
            turned_last = (Ry.cpu().numpy() @ last)
            wp = agent.data.x0[0, :, 0].cpu().numpy() if agent.data.x0.dim() == 4 else agent.data.x0.reshape(-1, 3).cpu().numpy()
            ahead = wp[min(9, len(wp) - 1)] - wp[0]
            path_yaw = math.degrees(math.atan2(ahead[0], ahead[2])) if np.hypot(ahead[0], ahead[2]) > 0.05 else None
            rows.append({
                "window": i, "seed": seed,
                "past_shift": wrap(yaw_of(turned_last) - yaw_of(last)),
                "first_shift_turned": wrap(yaw_of(first["past turned 90"]) - yaw_of(first["as is"])),
                "first_shift_dropped": wrap(yaw_of(first["past dropped"]) - yaw_of(first["as is"])),
                "handover_jump": wrap(yaw_of(first["as is"]) - yaw_of(last)),
                "first_vs_path": None if path_yaw is None else wrap(yaw_of(first["as is"]) - path_yaw),
                "last_vs_path": None if path_yaw is None else wrap(yaw_of(last) - path_yaw),
            })
            print(rows[-1], flush=True)
    med = lambda v: float(np.median(np.abs([x for x in v if x is not None]))) if any(x is not None for x in v) else None  # noqa: E731
    follow = [r["first_shift_turned"] / r["past_shift"] for r in rows if abs(r["past_shift"]) > 45]
    res = {"samples": len(rows),
           "past_shift_deg_median_abs": med([r["past_shift"] for r in rows]),
           "first_heading_moves_when_past_turned_deg_median_abs": med([r["first_shift_turned"] for r in rows]),
           "share_of_the_turn_followed_median": float(np.median(follow)) if follow else None,
           "first_heading_moves_when_past_dropped_deg_median_abs": med([r["first_shift_dropped"] for r in rows]),
           "handover_jump_deg_median_abs": med([r["handover_jump"] for r in rows]),
           "first_heading_vs_path_deg_median_abs": med([r["first_vs_path"] for r in rows]),
           "last_past_heading_vs_path_deg_median_abs": med([r["last_vs_path"] for r in rows]),
           "rows": rows}
    m = res["first_heading_moves_when_past_turned_deg_median_abs"]
    res["readout"] = "ignores the past facing" if m < 20 else "uses the past facing" if m > 60 else "in between"
    json.dump(res, open(os.path.join(out, "p0_past.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "rows"}, indent=1), flush=True)


if __name__ == "__main__":
    main()
