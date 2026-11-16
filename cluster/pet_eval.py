"""Scores a finished 4D-Animal reconstruction on its held-back frames, against the baseline of simply
showing the nearest frame that was used, and draws a turntable.

Run from the 4D-Animal code folder (it imports that code and reads its `external_data`):
    python pet_eval.py score EXPERIMENT_DIR      -> EXPERIMENT_DIR/eval/heldout.json, sheet_heldout.jpg
    python pet_eval.py orbit EXPERIMENT_DIR      -> EXPERIMENT_DIR/eval/turntable.mp4, sheet_orbit.jpg

The reconstruction is scored by the authors' own evaluators. The baseline uses the same formulas:
both pictures are black outside the animal's outline, as in those evaluators.
"""
import argparse
import json
import math
import os
import sys

import torch

sys.path.insert(0, os.getcwd())


def load(archive, frame_limit, device):
    from data.input_cop import get_input_cop_from_archive
    from model.inferencer import Inferencer

    ic = get_input_cop_from_archive(archive, frame_limit, device=device)
    inferencer = Inferencer(archive, use_archived_code=True)
    pose_model = inferencer.load_pose_model().to(device)
    texture_model = inferencer.load_texture_model().to(device)
    return ic, pose_model, texture_model


def turn_deg(a, b):
    """Angle of the rigid rotation that best maps point sets a onto b, (B, V, 3) each, in degrees."""
    a = a - a.mean(1, keepdim=True)
    b = b - b.mean(1, keepdim=True)
    u, _, vh = torch.linalg.svd(a.transpose(1, 2) @ b)
    d = torch.sign(torch.det(vh.transpose(1, 2) @ u.transpose(1, 2)))
    fix = torch.diag_embed(torch.stack([torch.ones_like(d), torch.ones_like(d), d], -1))
    r = vh.transpose(1, 2) @ fix @ u.transpose(1, 2)
    cos = ((r.diagonal(dim1=1, dim2=2).sum(-1) - 1) / 2).clamp(-1, 1)
    return torch.rad2deg(torch.acos(cos))


def score(args):
    from losses import iou, psnr
    from model.pose_models import compute_pose
    from model.texture_models.model_utils import render_images_wrapper
    from scene_optim.callbacks import CallbackEval
    from scene_optim.evaluators.lpips_evaluator import GLOBAL_LPIPS
    from torchvision.utils import save_image

    device = "cuda"
    ic, pose_model, texture_model = load(args.archive, args.frame_limit, device)
    images, masks = ic.images.to(device), ic.masks.to(device)
    cameras, dino = ic.cameras.to(device), ic.dino_feature.to(device)
    smal, renderer = ic.smal, ic.renderer
    used, held = ic.X_ind_train, ic.X_ind_test
    ts_all, ts_held = ic.X_ts, ic.X_ts_test

    # the reconstruction, with the authors' evaluators
    evaluator = CallbackEval(images, masks, cameras, smal, None, ic.image_size, device)
    avatar, _ = evaluator.call(pose_model, held, ts_held, dino[held], texture_model, return_individual=True)

    # the baseline: the nearest used frame in time, shown as is
    near = torch.stack([used[(used - t).abs().argmin()] for t in held])
    inside = masks.reshape(masks.shape[0], masks.shape[1], masks.shape[2]) > 0
    cut = torch.where(inside.unsqueeze(-1), images, torch.zeros_like(images))
    real, shown = cut[held], cut[near]
    with torch.no_grad():
        b_psnr = psnr(shown, real)
        b_psnrm = torch.stack(
            [psnr(real[i][inside[held[i]]].unsqueeze(0), shown[i][inside[held[i]]].unsqueeze(0))[0]
             for i in range(len(held))]
        )
        b_lpips = GLOBAL_LPIPS.to(device)(2 * real.permute(0, 3, 1, 2) - 1, 2 * shown.permute(0, 3, 1, 2) - 1).reshape(-1)
        b_iou = iou(inside[near].int(), inside[held].int()).reshape(-1)

        # pictures of the reconstruction, and how far each held-back view is turned from its nearest used view
        v_held, _, faces = compute_pose(smal, pose_model, held, ts_held, dino[held])
        drawn, drawn_mask = render_images_wrapper(
            texture_model, smal, renderer, pose_model, held, ts_held, dino[held], cameras[held], v_held, faces, None
        )
        drawn = drawn.clone()
        drawn[~(drawn_mask[..., 0] > 0)] = 0
        v_near, _, _ = compute_pose(smal, pose_model, near, ts_all[near], dino[near])
        cam_held = cameras[held].get_world_to_view_transform().transform_points(v_held)
        cam_near = cameras[near].get_world_to_view_transform().transform_points(v_near)
        turn = turn_deg(cam_near, cam_held)

    def mean(x):
        return float(torch.as_tensor(x).float().mean())

    def per_frame(x):
        return [round(float(v), 4) for v in torch.as_tensor(x).float().cpu()]

    out = {
        "experiment": os.path.abspath(args.archive),
        "frames_used": len(used),
        "frames_held_back": len(held),
        "frame_gap_max": int((near - held).abs().max()),
        "turn_deg_median": float(turn.median()),
        "turn_deg_max": float(turn.max()),
        "avatar": {"iou": mean(avatar["l_iou"]), "psnr": mean(avatar["l_psnr"]),
                   "psnr_mask": mean(avatar["l_psnrm"]), "lpips": mean(avatar["l_lpips"])},
        "baseline": {"iou": mean(b_iou), "psnr": mean(b_psnr), "psnr_mask": mean(b_psnrm), "lpips": mean(b_lpips)},
        "per_frame": {
            "frame": held.tolist(), "nearest_used": near.tolist(), "turn_deg": per_frame(turn),
            "avatar_iou": per_frame(avatar["l_iou"]), "avatar_psnr_mask": per_frame(avatar["l_psnrm"]),
            "avatar_lpips": per_frame(avatar["l_lpips"]),
            "baseline_iou": per_frame(b_iou), "baseline_psnr_mask": per_frame(b_psnrm),
            "baseline_lpips": per_frame(b_lpips),
        },
    }
    out["beats_baseline"] = {
        "lpips": out["avatar"]["lpips"] < out["baseline"]["lpips"],
        "iou": out["avatar"]["iou"] > out["baseline"]["iou"],
    }
    out["passes"] = all(out["beats_baseline"].values())
    folder = os.path.join(args.archive, "eval")
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, "heldout.json"), "w") as f:
        json.dump(out, f, indent=1)

    # rows: held-back photo, reconstruction, nearest used photo; eight held-back frames spread over the test set
    pick = torch.linspace(0, len(held) - 1, 8).round().long().to(device)
    rows = torch.cat([real[pick], drawn[pick], shown[pick]]).permute(0, 3, 1, 2).clamp(0, 1)
    save_image(rows, os.path.join(folder, "sheet_heldout.jpg"), nrow=8, padding=2)
    print(json.dumps({k: out[k] for k in ("frame_gap_max", "turn_deg_median", "avatar", "baseline", "passes")}))


def orbit(args):
    """A camera circling the animal, posed as in one frame that was used.

    "Up" is the normal of the plane through the video's camera centres (the person walks around the
    animal at roughly constant height); the circle keeps that frame's camera distance and height.
    """
    import imageio
    import numpy as np
    from model.pose_models import compute_pose
    from model.texture_models.model_utils import render_images_wrapper
    from pytorch3d.renderer import look_at_view_transform
    from torchvision.utils import save_image

    device = "cuda"
    ic, pose_model, texture_model = load(args.archive, args.frame_limit, device)
    cameras, dino = ic.cameras.to(device), ic.dino_feature.to(device)
    smal, renderer = ic.smal, ic.renderer
    used = ic.X_ind_train
    k = used[len(used) // 2].reshape(1)  # the pose of one frame that was used
    ts = ic.X_ts[k]
    with torch.no_grad():
        verts, _, faces = compute_pose(smal, pose_model, k, ts, dino[k])
    centre = verts[0].mean(0)

    # camera centres in the world: x_cam = x_world @ R + T, so C = -T @ R^T
    cams_c = -(cameras.T[:, None, :] @ cameras.R.transpose(1, 2))[:, 0]
    _, sv, vh = torch.linalg.svd(cams_c - cams_c.mean(0))
    up = vh[-1]
    cam_up = cameras.R[:, :, 1].mean(0)  # PyTorch3D cameras: +Y is up
    if float(up @ cam_up) < 0:
        up = -up
    if float(sv[1]) < 3 * float(sv[2]):  # camera path too short to define a plane
        up = cam_up / cam_up.norm()
    off = cams_c[int(k)] - centre
    height = float(off @ up)
    flat = off - height * up
    radius = float(flat.norm())
    u1 = flat / flat.norm()
    u2 = torch.linalg.cross(up, u1)
    n = args.views
    angles = torch.arange(n, device=device) * (2 * math.pi / n)
    eyes = centre + radius * (torch.cos(angles)[:, None] * u1 + torch.sin(angles)[:, None] * u2) + height * up
    rot, trans = look_at_view_transform(eye=eyes, at=centre[None].expand(n, -1), up=up[None].expand(n, -1), device=device)
    info = {"frame": int(k), "plane_singular_values": [round(float(v), 4) for v in sv],
            "elevation_deg": round(math.degrees(math.atan2(height, radius)), 1)}

    frames = []
    with torch.no_grad():
        for s in range(0, n, 8):
            idx = list(range(s, min(s + 8, n)))
            b = len(idx)
            cams = cameras[[int(k)] * b]
            cams.R, cams.T = rot[idx], trans[idx]
            kk = k.repeat(b)
            drawn, mask = render_images_wrapper(
                texture_model, smal, renderer, pose_model, kk, ts.repeat(b), dino[kk], cams,
                verts.expand(b, -1, -1), faces.expand(b, -1, -1), None,
            )
            m = (mask[..., :1] > 0).float()
            frames.append((drawn * m + 0.5 * (1 - m)).clamp(0, 1).cpu())
    frames = torch.cat(frames)
    folder = os.path.join(args.archive, "eval")
    os.makedirs(folder, exist_ok=True)
    imageio.mimsave(os.path.join(folder, "turntable.mp4"),
                    [(f.numpy() * 255).astype(np.uint8) for f in frames], fps=12)
    save_image(frames[:: max(1, n // 8)][:8].permute(0, 3, 1, 2), os.path.join(folder, "sheet_orbit.jpg"), nrow=8, padding=2)
    with open(os.path.join(folder, "turntable.json"), "w") as f:
        json.dump(info, f)
    print("turntable:", n, "views", json.dumps(info))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("what", choices=["score", "orbit"])
    p.add_argument("archive")
    p.add_argument("--frame_limit", type=int, default=200)
    p.add_argument("--views", type=int, default=48)
    a = p.parse_args()
    {"score": score, "orbit": orbit}[a.what](a)
