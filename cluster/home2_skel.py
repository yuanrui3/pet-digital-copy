"""Second cat at home, retarget (work/pets/PROTOCOL_CAT.md, "Second cat at home"): export the grey and white cat's own
skeleton and skinning from its Lab4D model, so Pikachu's motion can be carried over onto its own body
(home2_retarget.py). Its skeleton has the same 25 joints, tree and order as Agent-to-Sim's (both come from the same quad
URDF); only the rest pose, bone lengths and joint frames differ.

Run from the Lab4D code folder (env-lab4d, GPU):
  UV_NPZ=.../uv.npz python $D/tools/home2_skel.py --flagfile=logdir/LOG/opts.log --load_suffix latest --cat_out OUTDIR
Writes OUTDIR/grey_skel.npz (model units, canonical space): rest joint positions and rotations, parents, top-4
skinning weights at the UV mesh's vertices, and the bone motion (rest -> frame, 3x4) of every 2nd frame of its own
video, checked here against Lab4D's own warp.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cat_eval  # noqa: E402,F401  (defines --cat_out)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from absl import app  # noqa: E402

from lab4d.config import get_config  # noqa: E402
from lab4d.dataloader import data_utils  # noqa: E402
from lab4d.engine.model import dvr_model  # noqa: E402
from lab4d.engine.trainer import Trainer  # noqa: E402
from lab4d.utils.quat_transform import (dual_quaternion_inverse, dual_quaternion_mul,  # noqa: E402
                                        dual_quaternion_to_se3)
from lab4d.utils.skel_utils import fk_se3, shift_joints_to_bones  # noqa: E402


@torch.no_grad()
def main(_):
    opts = get_config()
    flagfile = [x.split("=", 1)[1] for x in sys.argv if x.startswith("--flagfile=")][0]
    opts["logroot"] = flagfile.rsplit("/", 2)[0]
    out_dir = opts["cat_out"]
    os.makedirs(out_dir, exist_ok=True)
    dev = "cuda"

    eval_dict = Trainer.construct_dataset_opts(opts, is_eval=True)
    evalloader = data_utils.eval_loader(eval_dict)
    data_info, _ = data_utils.get_data_info(evalloader)
    model = dvr_model(opts, data_info)
    ckpt = "%s/%s-%s/ckpt_%s.pth" % (opts["logroot"], opts["seqname"], opts["logname"], opts["load_suffix"])
    Trainer.load_checkpoint(ckpt, model)
    model.cuda().eval()
    field = model.fields.field_params["fg"]
    art = field.warp.articulation
    print("warp %s, articulation %s" % (type(field.warp).__name__, type(art).__name__), flush=True)
    n_raw = int(data_info["frame_info"]["frame_offset_raw"][-1])
    uvd = np.load(os.environ["UV_NPZ"])
    xyz = torch.tensor(uvd["pos"], dtype=torch.float32, device=dev)
    V = len(xyz)
    inst = torch.tensor([0], device=dev)

    # rest skeleton: joint angles at the mean time code with the canonical bone lengths (the rest half of
    # get_vals_and_mean); joints = forward kinematics + the learnt shift, as shift_joints_to_bones_dq does
    so3 = art.forward(art.time_embedding.get_mean_embedding(dev), None, return_so3=True).reshape(1, -1, 3)
    local = art.compute_rel_rest_joints().reshape(1, -1, 3)
    rt = fk_se3(local, so3, art.edges, to_dq=False)[0]  # (B,4,4)
    joints = rt[:, :3, 3] + art.shift.reshape(1, 3)
    rot = rt[:, :3, :3]
    rest = art.get_mean_vals()
    rest_se3 = dual_quaternion_to_se3(rest)[0]
    centres = shift_joints_to_bones(joints.clone()[None], art.edges)[0]
    print("rest skeleton check against get_mean_vals: centres off by %.2e, rotations by %.2e (model units: body %s)" % (
        float((centres - rest_se3[:, :3, 3]).abs().max()), float((rot - rest_se3[:, :3, :3]).abs().max()),
        np.ptp(uvd["pos"], 0).round(3).tolist()), flush=True)
    B = len(joints)
    parents = np.full(B, -1, dtype=np.int64)
    for idx, par in art.edges.items():
        parents[idx - 1] = par - 1

    # skinning weights at the rest pose, as SkinningWarp.forward computes them for forward warping
    rest_x = (rest[0][:, None, None].expand(1, 1, V, B, 4), rest[1][:, None, None].expand(1, 1, V, B, 4))
    skin, _ = field.warp.skinning_model(xyz[None, None], rest_x, None, inst)
    prob = skin.softmax(-1)[0, 0]
    w4, i4 = prob.topk(4, -1)
    cover = float(w4.sum(-1).mean())
    w4 = w4 / w4.sum(-1, keepdim=True)
    print("bones %d; top-4 weights cover %.4f of the weight on average" % (B, cover), flush=True)

    # its own video (every 2nd raw frame, as the page's Body view): rest -> frame bone motion, checked on a few frames
    frames = list(range(0, n_raw, 2))
    own, errs = [], []
    hom = torch.cat([xyz, torch.ones(V, 1, device=dev)], -1)
    for f in frames:
        fid = torch.tensor([f], dtype=torch.long, device=dev)
        t_art, rest_art = art.get_vals_and_mean(fid)
        se3 = dual_quaternion_to_se3(dual_quaternion_mul(t_art, dual_quaternion_inverse(rest_art)))[0, :, :3, :]
        own.append(se3.cpu().numpy())
        if f % 40 == 0:
            quat, trans = field.camera_mlp.get_vals(fid)
            rot_idn = torch.zeros_like(quat)
            rot_idn[..., 0] = 1.0
            ref = field.forward_warp(xyz[None, None], (rot_idn, torch.zeros_like(trans)), fid, inst,
                                     samples_dict={"t_articulation": t_art, "rest_articulation": rest_art})[0, 0]
            lbs = ((se3[i4] @ hom[:, None, :, None])[..., 0] * w4[..., None]).sum(1)
            err = (lbs - ref).norm(dim=-1)
            errs.append([f, float(err.median()), float(err.max())])
            print("frame %d: 4-bone blend vs Lab4D warp, median %.4f, max %.4f (model units)" % tuple(errs[-1]), flush=True)

    np.savez(os.path.join(out_dir, "grey_skel.npz"), joints=joints.cpu().numpy(), rot=rot.cpu().numpy(),
             parents=parents, skin_idx=i4.byte().cpu().numpy(), skin_w=w4.float().cpu().numpy(), skin_cover=cover,
             own=np.stack(own).astype(np.float32), own_frames=np.array(frames), check=np.array(errs))
    print("wrote grey_skel.npz: %d vertices, %d bones, %d own frames" % (V, B, len(frames)), flush=True)


if __name__ == "__main__":
    app.run(main)
