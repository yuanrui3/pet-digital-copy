"""Cat test C1, diagnostic without a rule (work/pets/PROTOCOL_CAT.md, Changes, 2026-11-15 10:00): is the drifting
appearance code the cause of the copy's darker, greyer coat on held-back frames?

Re-draws the held-back frames from the same trained model, same pose and camera, changing only the appearance code
(`appr_embedding`, a TimeMLP of the raw frame id) they get: unchanged / the nearest given frame's code in time /
in viewpoint / Lab4D's average code. No training. Run from the Lab4D code folder, after cat_eval.py,
cat_view_baseline.py and cat_colour.py:
  python $D/tools/cat_appr_swap.py --flagfile=logdir/LOG/opts.log --load_suffix latest --eval_res 256 --cat_out DIR
Writes DIR/appr_swap.json and DIR/sheet_appr_swap.jpg (rows: held-back photo, then one row per variant).
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cat_eval  # noqa: E402  (puts the Lab4D folder on sys.path and defines --cat_out)
import cat_colour  # noqa: E402

import cv2  # noqa: E402
import lpips  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from absl import app  # noqa: E402

from lab4d.config import get_config  # noqa: E402
from lab4d.dataloader import data_utils  # noqa: E402
from lab4d.engine.model import dvr_model  # noqa: E402
from lab4d.engine.trainer import Trainer  # noqa: E402


@torch.no_grad()
def main(_):
    opts = get_config()
    flagfile = [x.split("=", 1)[1] for x in sys.argv if x.startswith("--flagfile=")][0]
    opts["logroot"] = flagfile.rsplit("/", 2)[0]
    out_dir = opts["cat_out"]
    device = "cuda"
    js = {}
    for k in ("heldout", "view_baseline", "colour"):
        with open(os.path.join(out_dir, k + ".json")) as f:
            js[k] = json.load(f)
    h, v = js["heldout"]["per_frame"], js["view_baseline"]["per_frame"]

    eval_dict = Trainer.construct_dataset_opts(opts, is_eval=True)
    evalloader = data_utils.eval_loader(eval_dict)
    data_info, _ = data_utils.get_data_info(evalloader)
    model = dvr_model(opts, data_info)
    ckpt = "%s/%s-%s/ckpt_%s.pth" % (opts["logroot"], opts["seqname"], opts["logname"], opts["load_suffix"])
    Trainer.load_checkpoint(ckpt, model)
    model.cuda().eval()

    mapping = np.asarray(data_info["frame_info"]["frame_mapping"])
    a, b = [int(x) for x in opts["heldout_frames"].split(",")]
    held = np.arange(a, b)
    assert list(held) == h["frame"] == v["frame"]
    held_raw = [int(x) for x in mapping[held]]
    assert held_raw == h["raw_id"]

    # the held-back frames only, prepared exactly as cat_eval.py prepares every frame
    ref, batch = Trainer.load_batch(evalloader.dataset, held)
    for k, x in batch.items():
        batch[k] = torch.tensor(x, device=device)
    batch["crop2raw"][..., :2] *= opts["train_res"] / opts["eval_res"]
    hxy = Trainer.create_xy_grid(opts["eval_res"], device)
    batch["hxy"] = hxy[None].expand(len(batch["dataid"]), -1, -1)
    real_rgb = torch.tensor(ref["ref_rgb"], device=device).float()
    real_mask = torch.tensor(ref["ref_mask"], device=device)[..., 0].bool()
    vis = torch.tensor(ref["ref_vis2d"], device=device)[..., 0].bool()
    real_mask &= vis
    cut = real_rgb * real_mask[..., None]

    # swap the appearance code, and nothing else, for held-back frame ids
    emb = model.fields.field_params["fg"].appr_embedding
    orig_get = emb.get_vals
    state = {"sub": None, "mean": False, "hits": 0}
    held_raw_t = torch.tensor(held_raw, device=device)

    def get_vals(frame_id=None):
        if frame_id is None or (state["sub"] is None and not state["mean"]):
            return orig_get(frame_id)
        fid = torch.as_tensor(frame_id, device=device).long()
        is_held = torch.isin(fid, held_raw_t)
        state["hits"] += int(is_held.sum())
        if state["mean"]:
            out = orig_get(frame_id).clone()
            out[is_held] = emb.get_mean_vals().reshape(1, -1).to(out.dtype)
            return out
        new = fid.clone()
        for src, dst in state["sub"].items():
            new[fid == src] = dst
        return orig_get(new if torch.is_tensor(frame_id) else new.cpu().numpy())

    emb.get_vals = get_vals

    variants = {
        "unchanged": {"sub": None, "mean": False},
        "code_of_nearest_in_time": {"sub": dict(zip(held_raw, h["nearest_used_raw_id"])), "mean": False},
        "code_of_nearest_in_viewpoint": {"sub": dict(zip(held_raw, v["viewpoint_nearest_raw_id"])), "mean": False},
        "average_code": {"sub": None, "mean": True},
    }
    renders = {}
    for name, s in variants.items():
        state.update(s, hits=0)
        r = model.evaluate({k: x.clone() for k, x in batch.items()})
        if name != "unchanged":
            assert state["hits"] > 0, "no held-back frame id reached the appearance code: %s" % name
        renders[name] = (r["rgb"].float() * vis[..., None], (r["mask"][..., 0] > 0.5) & vis)
        print(name, "rendered; code swapped on", state["hits"], "rows", flush=True)

    lp = lpips.LPIPS(net="vgg", verbose=False).to(device)

    def u8(x):
        return (x.clamp(0, 1) * 255).round().byte().cpu().numpy()

    real_u8, rm = u8(cut), real_mask.cpu().numpy()
    truth = [cat_colour.cat_pixels(real_u8[i], rm[i]) for i in range(len(held))]
    res, per = {}, {}
    for name, (drawn, dmask) in renders.items():
        du8, dm = u8(drawn * dmask[..., None]), dmask.cpu().numpy()
        px = [cat_colour.cat_pixels(du8[i], dm[i]) for i in range(len(held))]
        md = [cat_colour.mean_diff(t, p) for t, p in zip(truth, px)]
        res[name] = {
            "mean_colour_diff": float(np.mean(md)),
            "colour_spread_dist": float(np.mean([cat_colour.spread_dist(t, p) for t, p in zip(truth, px)])),
            "lightness_L": float(np.mean([cat_colour.light_chroma(p)[0] for p in px])),
            "chroma": float(np.mean([cat_colour.light_chroma(p)[1] for p in px])),
            "lpips": float(lp(2 * cut.permute(0, 3, 1, 2) - 1, 2 * drawn.permute(0, 3, 1, 2) - 1).mean()),
            "iou": float(cat_eval.iou(dmask, real_mask).mean()),
            "psnr_mask": float(cat_eval.psnr_in_mask(real_rgb, drawn, real_mask).mean()),
        }
        per[name] = [round(x, 2) for x in md]

    un = res["unchanged"]
    drop = 1 - res["code_of_nearest_in_time"]["mean_colour_diff"] / un["mean_colour_diff"]
    out = {
        "diagnostic_without_a_rule": "C1's pass rule and verdict are in heldout.json",
        "unchanged_reproduces_earlier": {
            "lpips": abs(un["lpips"] - js["heldout"]["avatar"]["lpips"]) < 2e-3,
            "iou": abs(un["iou"] - js["heldout"]["avatar"]["iou"]) < 2e-3,
            "mean_colour_diff": abs(un["mean_colour_diff"]
                                    - js["colour"]["held_back_mean"]["copy"]["mean_colour_diff"]) < 0.1,
        },
        "variants": res,
        "readout_drop_in_mean_colour_diff_with_nearest_in_time_code": drop,
        "readout": ("the drifting appearance code is the main cause" if drop >= 0.5 else
                    "the appearance code is not the cause" if drop < 0.25 else
                    "the appearance code is part of the cause"),
        "context": {"photo_nearest_in_viewpoint_mean_colour_diff":
                    js["colour"]["held_back_mean"]["nearest_in_viewpoint"]["mean_colour_diff"],
                    "copy_on_given_frames_mean_colour_diff":
                    js["colour"]["given_views_context_copy"]["mean_colour_diff"],
                    "real_lightness_L": js["colour"]["held_back_mean"]["real"]["lightness_L"],
                    "real_chroma": js["colour"]["held_back_mean"]["real"]["chroma"]},
        "per_frame_mean_colour_diff": {"frame": [int(x) for x in held], "raw_id": held_raw, **per},
    }
    with open(os.path.join(out_dir, "appr_swap.json"), "w") as f:
        json.dump(out, f, indent=1)

    pick = np.unique(np.linspace(0, len(held) - 1, min(8, len(held))).round().astype(int))
    rows = [np.concatenate([real_u8[i] for i in pick], 1)]
    for name, (drawn, dmask) in renders.items():
        du8 = u8(drawn)
        rows.append(np.concatenate([du8[i] for i in pick], 1))
    cv2.imwrite(os.path.join(out_dir, "sheet_appr_swap.jpg"), np.concatenate(rows, 0)[:, :, ::-1])
    print(json.dumps({k: out[k] for k in ("unchanged_reproduces_earlier", "variants",
                                          "readout_drop_in_mean_colour_diff_with_nearest_in_time_code",
                                          "readout")}, indent=1))


if __name__ == "__main__":
    app.run(main)
