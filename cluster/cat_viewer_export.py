"""Exports a Lab4D cat model as a small animated, coloured mesh for a web viewer (demo, not a test).

Run from the Lab4D code folder (env-lab4d; DECIMATE_PY = a Python with open3d, e.g. env-a2s/bin/python):
  python $D/tools/cat_viewer_export.py --flagfile=logdir/LOG/opts.log --load_suffix latest --cat_out OUTDIR \
      [--viewer_grid 128 --viewer_faces 12000 --viewer_stride 2 --viewer_colour_frame 52]
Writes OUTDIR/viewer_data.bin (packed arrays) and OUTDIR/viewer_meta.json (their layout), plus rest.obj.

The cat is shown in its own body frame (articulation only; the camera, and so where the cat walks, is not included).
Colours are queried from the model's colour field at the mesh vertices, in three ways: with the appearance code of
one given frame, with Lab4D's average code, and with each frame's own code (the drifting colours the test measured).
"""
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cat_eval  # noqa: E402  (Lab4D folder on sys.path, --cat_out)
from cat_uv_bake_fill import fill_and_save  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import trimesh  # noqa: E402
from absl import app, flags  # noqa: E402

from lab4d.config import get_config  # noqa: E402
from lab4d.dataloader import data_utils  # noqa: E402
from lab4d.engine.model import dvr_model  # noqa: E402
from lab4d.engine.trainer import Trainer  # noqa: E402
from lab4d.utils.quat_transform import quaternion_to_matrix  # noqa: E402

flags.DEFINE_integer("viewer_grid", 128, "marching cubes grid")
flags.DEFINE_integer("viewer_faces", 12000, "faces after decimation")
flags.DEFINE_integer("viewer_stride", 2, "every n-th raw frame")
flags.DEFINE_integer("viewer_colour_frame", 52, "raw frame whose appearance code gives the fixed colours")


def u8(x):
    return (x.clamp(0, 1) * 255).round().byte().cpu().numpy()


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
    n_raw = int(data_info["frame_info"]["frame_offset_raw"][-1])
    mapping = np.asarray(data_info["frame_info"]["frame_mapping"])

    # rest mesh, decimated with open3d in another environment; or, with UV_NPZ, the welded UV-mapped mesh that
    # cat_uv_bake.py made from an earlier export's rest.obj (then the colour field is also baked into textures)
    uvd = np.load(os.environ["UV_NPZ"]) if os.environ.get("UV_NPZ") else None
    if uvd is not None:
        mesh = trimesh.Trimesh(vertices=uvd["pos"], faces=uvd["faces"], process=False)
        print("UV-mapped mesh from %s: %d vertices, %d faces" % (os.environ["UV_NPZ"], len(mesh.vertices),
                                                                len(mesh.faces)), flush=True)
    else:
      mesh = field.extract_canonical_mesh(grid_size=opts["viewer_grid"], level=0.0, inst_id=0,
                                          use_visibility=False, use_extend_aabb=False)
      full = os.path.join(out_dir, "rest_full.ply")
      small = os.path.join(out_dir, "rest.obj")
      mesh.export(full)
      code = ("import open3d as o3d,sys; m=o3d.io.read_triangle_mesh(sys.argv[1]);"
              "m=m.simplify_quadric_decimation(int(sys.argv[3])); m.remove_unreferenced_vertices();"
              "m.remove_degenerate_triangles(); o3d.io.write_triangle_mesh(sys.argv[2], m)")
      subprocess.run([os.environ["DECIMATE_PY"], "-c", code, full, small, str(opts["viewer_faces"])], check=True)
      mesh = trimesh.load(small, process=False)
      print("rest mesh: full %d faces -> %d vertices, %d faces" % (len(trimesh.load(full).faces), len(mesh.vertices),
                                                                   len(mesh.faces)), flush=True)
    xyz = torch.tensor(mesh.vertices, dtype=torch.float32, device=dev)
    inst = torch.tensor([0], device=dev)

    frames = list(range(0, n_raw, opts["viewer_stride"]))
    pos, cols, down = [], [], []
    for f in frames:
        fid = torch.tensor([f], dtype=torch.long, device=dev)
        quat, trans = field.camera_mlp.get_vals(fid)
        rot_idn = torch.zeros_like(quat)
        rot_idn[..., 0] = 1.0
        sd = {}
        sd["t_articulation"], sd["rest_articulation"] = field.warp.articulation.get_vals_and_mean(fid)
        x_t = field.forward_warp(xyz[None, None], (rot_idn, torch.zeros_like(trans)), fid, inst, samples_dict=sd)
        pos.append(x_t[0, 0].float().cpu().numpy())
        rgb, _ = field.forward(xyz[None, :, None], dir=torch.zeros_like(xyz)[None, :, None], frame_id=fid,
                               inst_id=inst, get_density=False)
        cols.append(u8(rgb[0, :, 0]))
        r = quaternion_to_matrix(quat)[0].float().cpu().numpy()  # object -> camera
        down.append(r.T @ np.array([0.0, 1.0, 0.0]))  # camera's image-down direction in the body frame
    pos = np.stack(pos)  # (T,V,3)
    cols = np.stack(cols)  # (T,V,3)

    def colours_with(fid_val=None, mean=False):
        emb = field.appr_embedding
        orig = emb.get_vals
        if mean:
            emb.get_vals = lambda frame_id=None: emb.get_mean_vals().reshape(1, -1)
        fid = torch.tensor([fid_val if fid_val is not None else 0], dtype=torch.long, device=dev)
        rgb, _ = field.forward(xyz[None, :, None], dir=torch.zeros_like(xyz)[None, :, None], frame_id=fid,
                               inst_id=inst, get_density=False)
        emb.get_vals = orig
        return u8(rgb[0, :, 0])

    col_fixed = colours_with(opts["viewer_colour_frame"])
    col_mean = colours_with(mean=True)
    textures = {}
    if uvd is not None:  # the same two colourings, at every texel
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from PIL import Image  # noqa: F401  (fill_and_save needs it)
        tp = torch.tensor(uvd["texel_pos"], dtype=torch.float32, device=dev)

        def texel_colours(fid_val=None, mean=False):
            emb = field.appr_embedding
            orig = emb.get_vals
            if mean:
                emb.get_vals = lambda frame_id=None: emb.get_mean_vals().reshape(1, -1)
            fid = torch.tensor([fid_val if fid_val is not None else 0], dtype=torch.long, device=dev)
            out = []
            for k in range(0, len(tp), 200000):
                p = tp[k:k + 200000]
                rgb, _ = field.forward(p[None, :, None], dir=torch.zeros_like(p)[None, :, None], frame_id=fid,
                                       inst_id=inst, get_density=False)
                out.append(rgb[0, :, 0].clamp(0, 1))
            emb.get_vals = orig
            return torch.cat(out).cpu().numpy()

        size = int(uvd["size"])
        for name, kw in (("cat_tex_fixed.jpg", {"fid_val": opts["viewer_colour_frame"]}), ("cat_tex_mean.jpg", {"mean": True})):
            fill_and_save(texel_colours(**kw), uvd["texel_rc"], size, os.path.join(out_dir, name))
        textures = {"fixed": "cat_tex_fixed.jpg", "mean": "cat_tex_mean.jpg"}
        print("textures baked: %d texels, %d px" % (len(tp), size), flush=True)
    down = np.mean(down, 0)
    down /= np.linalg.norm(down)

    lo, hi = pos.reshape(-1, 3).min(0), pos.reshape(-1, 3).max(0)
    q = np.round((pos - lo) / (hi - lo) * 65535).astype(np.uint16)
    faces = np.asarray(mesh.faces, dtype=np.uint16 if len(mesh.vertices) < 65536 else np.uint32)
    blobs = [("faces", faces), ("pos", q), ("col_fixed", col_fixed), ("col_mean", col_mean), ("col_frame", cols)]
    if uvd is not None:
        blobs.append(("uv", uvd["uv"].astype(np.float32)))
    meta = {"vertices": int(len(mesh.vertices)), "faces": int(len(mesh.faces)), "frames": len(frames),
            "raw_frames": frames, "raw_total": n_raw, "lo": lo.tolist(), "hi": hi.tolist(), "down": down.tolist(),
            "colour_frame": opts["viewer_colour_frame"],
            "used_raw_ids": [int(x) for x in mapping], "heldout_frames": opts.get("heldout_frames", ""),
            "textures": textures,
            "blobs": []}
    off = 0
    with open(os.path.join(out_dir, "viewer_data.bin"), "wb") as fh:
        for name, arr in blobs:
            b = np.ascontiguousarray(arr).tobytes()
            meta["blobs"].append({"name": name, "dtype": str(arr.dtype), "shape": list(arr.shape), "offset": off,
                                  "bytes": len(b)})
            fh.write(b)
            pad = (-len(b)) % 4
            fh.write(b"\0" * pad)
            off += len(b) + pad
    with open(os.path.join(out_dir, "viewer_meta.json"), "w") as fh:
        json.dump(meta, fh)
    print("wrote %d bytes for %d frames" % (off, len(frames)))


if __name__ == "__main__":
    app.run(main)
