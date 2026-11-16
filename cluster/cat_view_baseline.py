"""Cat test C1, extra comparison reported without a rule (work/pets/PROTOCOL_CAT.md, Changes, 2026-11-15 07:40):
the used photo nearest in viewpoint, instead of the nearest in time. Same data, crop, mask and measures as
cat_eval.py (whose functions it reuses); nothing is rendered: the copy's scores are read from heldout.json.

Run from the Lab4D code folder, after cat_eval.py:
  python $D/tools/cat_view_baseline.py --flagfile=logdir/LOG/opts.log --load_suffix latest --eval_res 256 --cat_out DIR
Writes DIR/view_baseline.json and DIR/sheet_view_baseline.jpg (rows: held-back photo, copy, nearest used photo in
time, nearest used photo in viewpoint).
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cat_eval  # noqa: E402  (puts the Lab4D folder on sys.path and defines --cat_out)

import cv2  # noqa: E402
import lpips  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from absl import app  # noqa: E402

from lab4d.config import get_config  # noqa: E402
from lab4d.dataloader import data_utils  # noqa: E402
from lab4d.engine.model import dvr_model  # noqa: E402
from lab4d.engine.trainer import Trainer  # noqa: E402

iou, psnr, psnr_in_mask, rot_deg = cat_eval.iou, cat_eval.psnr, cat_eval.psnr_in_mask, cat_eval.rot_deg


@torch.no_grad()
def main(_):
    opts = get_config()
    flagfile = [x.split("=", 1)[1] for x in sys.argv if x.startswith("--flagfile=")][0]
    opts["logroot"] = flagfile.rsplit("/", 2)[0]
    out_dir = opts["cat_out"]
    device = "cuda"
    with open(os.path.join(out_dir, "heldout.json")) as f:
        h = json.load(f)

    # data and model exactly as cat_eval.py loads them (the model only for its fitted cameras)
    eval_dict = Trainer.construct_dataset_opts(opts, is_eval=True)
    evalloader = data_utils.eval_loader(eval_dict)
    data_info, _ = data_utils.get_data_info(evalloader)
    model = dvr_model(opts, data_info)
    ckpt = "%s/%s-%s/ckpt_%s.pth" % (opts["logroot"], opts["seqname"], opts["logname"], opts["load_suffix"])
    Trainer.load_checkpoint(ckpt, model)
    model.cuda().eval()

    mapping = np.asarray(data_info["frame_info"]["frame_mapping"])
    n = len(mapping)
    a, b = [int(x) for x in opts["heldout_frames"].split(",")]
    fids = np.arange(n - 1)
    ref, _ = Trainer.load_batch(evalloader.dataset, fids)
    real_rgb = torch.tensor(ref["ref_rgb"], device=device).float()
    real_mask = torch.tensor(ref["ref_mask"], device=device)[..., 0].bool()
    vis = torch.tensor(ref["ref_vis2d"], device=device)[..., 0].bool()
    real_mask &= vis
    cut = real_rgb * real_mask[..., None]

    held = torch.arange(a, b, device=device)
    used = torch.tensor([i for i in range(n - 1) if not (a <= i < b)], device=device)
    raw = torch.tensor(mapping, device=device)
    near_time = torch.stack([used[(raw[used] - raw[t]).abs().argmin()] for t in held])

    cams = model.fields.get_cameras(frame_id=mapping)["fg"][:, :3, :3].float().cpu()
    uc = used.cpu()
    ang = torch.stack([rot_deg(cams[t][None].expand(len(uc), 3, 3), cams[uc]) for t in held.cpu()])  # (held, used)
    near_view = used[ang.argmin(1).to(device)]
    view_deg = ang.min(1).values
    time_deg = rot_deg(cams[held.cpu()], cams[near_time.cpu()])

    lp = lpips.LPIPS(net="vgg", verbose=False).to(device)

    def lpips_of(x, y):
        return lp(2 * x.permute(0, 3, 1, 2) - 1, 2 * y.permute(0, 3, 1, 2) - 1).reshape(-1)

    def scores(frames_shown):
        return {
            "iou": iou(real_mask[frames_shown], real_mask[held]),
            "psnr": psnr(cut[held], cut[frames_shown]),
            "psnr_mask": psnr_in_mask(real_rgb[held], cut[frames_shown], real_mask[held]),
            "lpips": lpips_of(cut[held], cut[frames_shown]),
        }

    by_time, by_view = scores(near_time), scores(near_view)

    def mean(d):
        return {k: float(v.mean()) for k, v in d.items()}

    def pf(x):
        return [round(float(v), 4) for v in x]

    copy_lpips = np.array(h["per_frame"]["avatar_lpips"])
    reproduces = all(abs(mean(by_time)[k] - h["baseline"][k]) < 1e-3 for k in ("iou", "psnr_mask", "lpips"))
    out = {
        "reported_without_a_rule": "C1's pass rule and verdict use the nearest photo in time (heldout.json)",
        "copy": h["avatar"],
        "nearest_in_time": mean(by_time),
        "nearest_in_time_reproduces_heldout_json": reproduces,
        "nearest_in_viewpoint": mean(by_view),
        "copy_better_than_nearest_in_viewpoint": {
            "lpips": h["avatar"]["lpips"] < mean(by_view)["lpips"],
            "iou": h["avatar"]["iou"] > mean(by_view)["iou"],
            "psnr_mask": h["avatar"]["psnr_mask"] > mean(by_view)["psnr_mask"],
        },
        "frames_copy_lpips_better_than_viewpoint_photo": int((copy_lpips < by_view["lpips"].cpu().numpy()).sum()),
        "frames_held_back": len(held),
        "turn_deg_nearest_in_time_median": float(time_deg.median()),
        "turn_deg_nearest_in_viewpoint_median": float(view_deg.median()),
        "turn_deg_nearest_in_viewpoint_max": float(view_deg.max()),
        "per_frame": {
            "frame": held.tolist(), "raw_id": raw[held].tolist(),
            "viewpoint_nearest": near_view.tolist(), "viewpoint_nearest_raw_id": raw[near_view].tolist(),
            "turn_deg_viewpoint": pf(view_deg), "turn_deg_time": pf(time_deg),
            "copy_lpips": pf(copy_lpips), "viewpoint_lpips": pf(by_view["lpips"]),
            "viewpoint_iou": pf(by_view["iou"]), "viewpoint_psnr_mask": pf(by_view["psnr_mask"]),
        },
    }
    with open(os.path.join(out_dir, "view_baseline.json"), "w") as f:
        json.dump(out, f, indent=1)

    # sheet: the same 8 frames as sheet_heldout.jpg, plus the viewpoint photo as a fourth row
    fr = np.load(os.path.join(out_dir, "frames.npz"))
    pick = np.unique(np.linspace(0, len(held) - 1, min(8, len(held))).round().astype(int))
    hc, tc, vc = held.cpu().numpy(), near_time.cpu().numpy(), near_view.cpu().numpy()
    cut_u8 = (cut.clamp(0, 1) * 255).round().byte().cpu().numpy()
    rows = [np.concatenate([imgs[i] for i in pick], 1)
            for imgs in (cut_u8[hc], fr["drawn"][hc], cut_u8[tc], cut_u8[vc])]
    cv2.imwrite(os.path.join(out_dir, "sheet_view_baseline.jpg"), np.concatenate(rows, 0)[:, :, ::-1])

    print(json.dumps({k: out[k] for k in ("copy", "nearest_in_time", "nearest_in_time_reproduces_heldout_json",
                                          "nearest_in_viewpoint", "copy_better_than_nearest_in_viewpoint",
                                          "frames_copy_lpips_better_than_viewpoint_photo",
                                          "turn_deg_nearest_in_viewpoint_median",
                                          "turn_deg_nearest_in_viewpoint_max")}, indent=1))


if __name__ == "__main__":
    app.run(main)
