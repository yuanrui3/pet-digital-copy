"""The rebuild from this cat's close-up clips (work/pets/PROTOCOL_CAT.md): export the trained 18-clip model for the
checks R1 to R4. The rest mesh (grid 256, decimated to 44,000 faces with open3d in another environment), and, for every
clip, the mesh posed at each kept frame in that frame's camera (the model's own object-to-camera transform), with the
frame's intrinsics, so it can be drawn into the video's own pixels.

Run from the Lab4D code folder (env-lab4d, GPU; DECIMATE_PY = a Python with open3d):
  python $D/tools/pika18_export.py --flagfile=logdir/LOG/opts.log --load_suffix latest --cat_out OUTDIR
Writes OUTDIR/rest.obj and OUTDIR/cam_<clip folder>.npz (pos: kept frames x V x 3 in camera space; faces; ks: fx, fy,
px, py in raw pixels; raw: each kept frame's raw number within its clip)."""
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cat_eval  # noqa: E402,F401  (defines --cat_out)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import trimesh  # noqa: E402
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
    fi = data_info["frame_info"]
    mapping = np.asarray(fi["frame_mapping"]).astype(np.int64)        # kept frame (all clips in a row) -> raw number
    off, off_raw = np.asarray(fi["frame_offset"]).astype(np.int64), np.asarray(fi["frame_offset_raw"]).astype(np.int64)
    print("clips %d; kept frames %d; raw frames %d" % (len(off) - 1, len(mapping), off_raw[-1]), flush=True)

    full, small = os.path.join(out_dir, "rest_full.ply"), os.path.join(out_dir, "rest.obj")
    mesh = field.extract_canonical_mesh(grid_size=256, level=0.0, inst_id=0, use_visibility=False, use_extend_aabb=False)
    mesh.export(full)
    code = ("import open3d as o3d,sys; m=o3d.io.read_triangle_mesh(sys.argv[1]);"
            "m=m.simplify_quadric_decimation(int(sys.argv[3])); m.remove_unreferenced_vertices();"
            "m.remove_degenerate_triangles(); o3d.io.write_triangle_mesh(sys.argv[2], m)")
    subprocess.run([os.environ["DECIMATE_PY"], "-c", code, full, small, "44000"], check=True)
    mesh = trimesh.load(small, process=False)
    print("rest mesh: %d vertices, %d faces" % (len(mesh.vertices), len(mesh.faces)), flush=True)
    xyz = torch.tensor(np.asarray(mesh.vertices), dtype=torch.float32, device=dev)

    seqs = [p.rstrip("/").split("/")[-1] for p in _img_paths(opts)]
    for v in range(len(off) - 1):
        inst = torch.tensor([v], device=dev)
        pos, ks, raw = [], [], []
        for k in range(off[v], off[v + 1]):
            f = int(mapping[k])
            fid = torch.tensor([f], dtype=torch.long, device=dev)
            quat, trans = field.camera_mlp.get_vals(fid)
            sd = {}
            sd["t_articulation"], sd["rest_articulation"] = field.warp.articulation.get_vals_and_mean(fid)
            x_t = field.forward_warp(xyz[None, None], (quat, trans), fid, inst, samples_dict=sd)
            pos.append(x_t[0, 0].float().cpu().numpy())
            ks.append(model.intrinsics.get_vals(fid)[0].float().cpu().numpy())
            raw.append(f - int(off_raw[v]))
        name = seqs[v] if v < len(seqs) else "clip%02d" % v
        np.savez(os.path.join(out_dir, "cam_%s.npz" % name), pos=np.stack(pos).astype(np.float32),
                 faces=np.asarray(mesh.faces).astype(np.int32), ks=np.stack(ks).astype(np.float32), raw=np.array(raw))
        print("  %s: %d kept frames, depth median %.2f" % (name, len(pos), float(np.median(np.stack(pos)[..., 2]))), flush=True)


def _img_paths(opts):
    import configparser
    cfg = configparser.RawConfigParser()
    cfg.read("database/configs/%s.config" % opts["seqname"])
    out = []
    k = 0
    while cfg.has_section("data_%d" % k):
        out.append(cfg.get("data_%d" % k, "img_path"))
        k += 1
    return out


if __name__ == "__main__":
    app.run(main)
