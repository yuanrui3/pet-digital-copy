"""Scores a Lab4D cat reconstruction on its held-back frames (work/pets/PROTOCOL_CAT.md).

Run from the Lab4D code folder, after training:
  python $D/tools/cat_eval.py --flagfile=logdir/LOG/opts.log --load_suffix latest --eval_res 256 --cat_out DIR

Writes DIR/heldout.json (same keys as the dog test's) and DIR/sheet_heldout.jpg (rows: held-back photo,
reconstruction, nearest used photo). The reconstruction is drawn by Lab4D itself in its own 256 px crop of
each frame; the baseline is the nearest used frame in time, cut out by its own mask and shown as is.
"""
import json
import os
import sys

cwd = os.getcwd()
if cwd not in sys.path:
    sys.path.insert(0, cwd)

import cv2
import lpips
import numpy as np
import torch
from absl import app, flags

import lab4d.render  # noqa: F401  (defines inst_id, render_res, ... used by construct_test_model)
from lab4d.config import get_config
from lab4d.dataloader import data_utils
from lab4d.engine.model import dvr_model
from lab4d.engine.trainer import Trainer

flags.DEFINE_string("cat_out", "", "output folder for heldout.json and the picture sheet")
SMOOTH = 1e-6  # as in 4D-Animal's losses/loss_utils.py


def iou(a, b):
    """a, b: (M,H,W) bool -> (M,) as 4D-Animal's iou"""
    inter = (a & b).float().sum((1, 2))
    union = (a | b).float().sum((1, 2))
    return (inter + SMOOTH) / (union + SMOOTH)


def psnr(x_gt, y):
    """as 4D-Animal's psnr, max 1"""
    mse = torch.mean((x_gt - y) ** 2, dim=list(range(1, x_gt.dim())))
    return 20 * torch.log10(1.0 / (torch.sqrt(mse) + 1e-6))


def psnr_in_mask(real, shown, inside):
    return torch.stack([psnr(real[i][inside[i]][None], shown[i][inside[i]][None])[0] for i in range(len(real))])


def rot_deg(ra, rb):
    """angle (degrees) of the rotation taking rb to ra, per row; ra, rb (M,3,3)"""
    rel = ra @ rb.transpose(1, 2)
    cos = ((rel.diagonal(dim1=1, dim2=2).sum(-1) - 1) / 2).clamp(-1, 1)
    return torch.rad2deg(torch.acos(cos))


@torch.no_grad()
def main(_):
    opts = get_config()
    flagfile = [x.split("=", 1)[1] for x in sys.argv if x.startswith("--flagfile=")][0]
    opts["logroot"] = flagfile.rsplit("/", 2)[0]  # as lab4d/render.py does
    out_dir = opts["cat_out"]
    os.makedirs(out_dir, exist_ok=True)
    device = "cuda"

    # model and data, as Trainer.construct_test_model does
    eval_dict = Trainer.construct_dataset_opts(opts, is_eval=True)
    evalloader = data_utils.eval_loader(eval_dict)
    data_info, _ = data_utils.get_data_info(evalloader)
    model = dvr_model(opts, data_info)
    ckpt = "%s/%s-%s/ckpt_%s.pth" % (opts["logroot"], opts["seqname"], opts["logname"], opts["load_suffix"])
    Trainer.load_checkpoint(ckpt, model)
    model.cuda().eval()

    mapping = np.asarray(data_info["frame_info"]["frame_mapping"])  # filtered frame -> raw frame id
    n = len(mapping)
    a, b = [int(x) for x in opts["heldout_frames"].split(",")]
    fids = np.arange(n - 1)  # the eval set holds pairs, so the last filtered frame has no entry
    ref, batch = Trainer.load_batch(evalloader.dataset, fids)
    for k, v in batch.items():
        batch[k] = torch.tensor(v, device=device)
    batch["crop2raw"][..., :2] *= opts["train_res"] / opts["eval_res"]
    hxy = Trainer.create_xy_grid(opts["eval_res"], device)
    batch["hxy"] = hxy[None].expand(len(batch["dataid"]), -1, -1)
    rendered = model.evaluate(batch)  # (n-1, H, W, x); colours already multiplied by the drawn mask

    real_rgb = torch.tensor(ref["ref_rgb"], device=device).float()
    real_mask = torch.tensor(ref["ref_mask"], device=device)[..., 0].bool()
    vis = torch.tensor(ref["ref_vis2d"], device=device)[..., 0].bool()  # pixels inside the video frame
    real_mask &= vis
    cut = real_rgb * real_mask[..., None]
    drawn_mask = (rendered["mask"][..., 0] > 0.5) & vis
    drawn = rendered["rgb"].float() * vis[..., None]

    held = torch.arange(a, b, device=device)
    used = torch.tensor([i for i in range(n - 1) if not (a <= i < b)], device=device)
    raw = torch.tensor(mapping, device=device)
    near = torch.stack([used[(raw[used] - raw[t]).abs().argmin()] for t in held])

    lp = lpips.LPIPS(net="vgg", verbose=False).to(device)

    def lpips_of(x, y):
        return lp(2 * x.permute(0, 3, 1, 2) - 1, 2 * y.permute(0, 3, 1, 2) - 1).reshape(-1)

    def scores(shown_rgb, shown_mask, frames):
        return {
            "iou": iou(shown_mask, real_mask[frames]),
            "psnr": psnr(cut[frames], shown_rgb),
            "psnr_mask": psnr_in_mask(real_rgb[frames], shown_rgb, real_mask[frames]),
            "lpips": lpips_of(cut[frames], shown_rgb),
        }

    avatar = scores(drawn[held], drawn_mask[held], held)
    baseline = scores(cut[near], real_mask[near], held)
    given = scores(drawn[used], drawn_mask[used], used)  # context: views whose colours were given

    # object-to-camera rotation of every filtered frame (numpy ids, as lab4d/render.py passes them)
    cams = model.fields.get_cameras(frame_id=mapping)["fg"][:, :3, :3].float().cpu()
    turn = rot_deg(cams[held.cpu()], cams[near.cpu()])

    def mean(x):
        return float(x.mean())

    def per_frame(x):
        return [round(float(v), 4) for v in x]

    out = {
        "experiment": ckpt,
        "frames_filtered": n,
        "frames_used": len(used),
        "frames_held_back": len(held),
        "held_back_ordinals": [a, b - 1],
        "held_back_raw_ids": [int(mapping[a]), int(mapping[b - 1])],
        "frame_gap_max_raw": int((raw[near] - raw[held]).abs().max()),
        "turn_deg_median": float(turn.median()),
        "turn_deg_max": float(turn.max()),
        "avatar": {k: mean(v) for k, v in avatar.items()},
        "baseline": {k: mean(v) for k, v in baseline.items()},
        "given_views_context": {k: mean(v) for k, v in given.items()},
        "per_frame": {
            "frame": held.tolist(), "raw_id": raw[held].tolist(), "nearest_used": near.tolist(),
            "nearest_used_raw_id": raw[near].tolist(), "turn_deg": per_frame(turn),
            "avatar_iou": per_frame(avatar["iou"]), "avatar_psnr_mask": per_frame(avatar["psnr_mask"]),
            "avatar_lpips": per_frame(avatar["lpips"]),
            "baseline_iou": per_frame(baseline["iou"]), "baseline_psnr_mask": per_frame(baseline["psnr_mask"]),
            "baseline_lpips": per_frame(baseline["lpips"]),
        },
    }
    out["beats_baseline"] = {
        "lpips": out["avatar"]["lpips"] < out["baseline"]["lpips"],
        "iou": out["avatar"]["iou"] > out["baseline"]["iou"],
    }
    out["passes"] = out["beats_baseline"]["lpips"] and out["beats_baseline"]["iou"]
    with open(os.path.join(out_dir, "heldout.json"), "w") as f:
        json.dump(out, f, indent=1)

    # picture sheet: up to 8 held-back frames spread over the block
    pick = np.unique(np.linspace(0, len(held) - 1, min(8, len(held))).round().astype(int))
    rows = []
    for imgs in (cut[held], drawn[held], cut[near]):
        rows.append(np.concatenate([imgs[i].cpu().numpy() for i in pick], 1))
    sheet = (np.clip(np.concatenate(rows, 0), 0, 1) * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(out_dir, "sheet_heldout.jpg"), sheet[:, :, ::-1])

    # every frame's real cut-out and drawing, for the identity check (cat_identity.py)
    def u8(x):
        return (x.clamp(0, 1) * 255).round().byte().cpu().numpy()

    np.savez_compressed(
        os.path.join(out_dir, "frames.npz"),
        real=u8(cut), real_mask=real_mask.cpu().numpy(),
        drawn=u8(drawn * drawn_mask[..., None]), drawn_mask=drawn_mask.cpu().numpy(),
        held=held.cpu().numpy(), used=used.cpu().numpy(), raw=mapping,
    )

    print(json.dumps({k: out[k] for k in ("avatar", "baseline", "given_views_context", "passes")}, indent=1))
    print("held back:", out["frames_held_back"], "frames; turn median %.1f max %.1f deg"
          % (out["turn_deg_median"], out["turn_deg_max"]))


if __name__ == "__main__":
    app.run(main)
