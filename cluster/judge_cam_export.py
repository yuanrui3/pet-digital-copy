"""The judge without ratings, check T2 (work/pets/PROTOCOL_CAT.md, "Check T2, corrected"): a Lab4D cat model posed at
every kept frame of its own video, in that frame's camera (the model's own optimised object-to-camera transform), with
the frame's intrinsics, so it can be drawn into the video's own pixels.

Run from the Lab4D code folder (env-lab4d, GPU):
  UV_NPZ=.../uv.npz python $D/tools/judge_cam_export.py --flagfile=logdir/LOG/opts.log --load_suffix latest --cat_out OUTDIR
Writes OUTDIR/recon_cam.npz: pos (kept frames, V, 3) in camera space (model units), faces, ks (kept frames, 4: fx, fy,
px, py in raw pixels), raw (each kept frame's raw number)."""
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
    mapping = np.asarray(data_info["frame_info"]["frame_mapping"]).astype(np.int64)  # kept frame -> raw number
    uvd = np.load(os.environ["UV_NPZ"])
    xyz = torch.tensor(uvd["pos"], dtype=torch.float32, device=dev)
    inst = torch.tensor([0], device=dev)
    pos, ks = [], []
    for f in mapping:
        fid = torch.tensor([int(f)], dtype=torch.long, device=dev)
        quat, trans = field.camera_mlp.get_vals(fid)
        sd = {}
        sd["t_articulation"], sd["rest_articulation"] = field.warp.articulation.get_vals_and_mean(fid)
        x_t = field.forward_warp(xyz[None, None], (quat, trans), fid, inst, samples_dict=sd)
        pos.append(x_t[0, 0].float().cpu().numpy())
        ks.append(model.intrinsics.get_vals(fid)[0].float().cpu().numpy())
    pos, ks = np.stack(pos), np.stack(ks)
    print("kept frames %d (raw %d..%d); camera-space depth median %.2f; ks[0] %s" % (
        len(mapping), mapping[0], mapping[-1], float(np.median(pos[..., 2])), np.round(ks[0], 1)), flush=True)
    np.savez(os.path.join(out_dir, "recon_cam.npz"), pos=pos.astype(np.float32), faces=uvd["faces"].astype(np.int32),
             ks=ks.astype(np.float32), raw=mapping)


if __name__ == "__main__":
    app.run(main)
