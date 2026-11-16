"""Exports Agent-to-Sim's own cat (reconstructed from the month of home videos) as a skinned, coloured mesh, plus the
bone motion of every recorded frame, so the web viewer can move the cat's body exactly as recorded (demo, not a test).

Run from agent2sim/gdmdm (env-a2s, PYTHONPATH=agent2sim/lab4d, GPU):
  DECIMATE_PY=python-with-open3d python a2s_cat_export.py LOGNAME MOTION_PKL OUT_DIR [FACES] [ROLLOUTS_JSON]
Writes OUT_DIR/a2s_cat.bin + a2s_cat.json: rest mesh (metres), colours (average appearance code), top-4 skinning
weights, and per recorded frame the 25 bone transforms (rest -> that frame, 3x4, metres), in the order of
a2s_room_export.py's paths. With ROLLOUTS_JSON (from b1_rollout.py, joint angles included) it also writes the bone
transforms of the copy's own walks (OUT_DIR/a2s_walks.bin + .json).
Checks itself: 4-bone linear blending against Lab4D's own dual-quaternion warp, on a few recorded frames.
"""
import json
import os
import pickle
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import torch  # noqa: E402
import trimesh  # noqa: E402

from a2s_room_export import pack  # noqa: E402
from b1_split import same_video  # noqa: E402
from lab4d.config import load_flags_from_file  # noqa: E402
from lab4d.engine.trainer import Trainer  # noqa: E402
from lab4d.utils.quat_transform import dual_quaternion_inverse, dual_quaternion_mul, dual_quaternion_to_se3  # noqa: E402


@torch.no_grad()
def main(logname, pkl_path, out_dir, faces=30000, rollouts=None):
    os.makedirs(out_dir, exist_ok=True)
    dev = "cuda"
    opts = load_flags_from_file("logdir/%s/opts.log" % logname)
    opts["load_suffix"] = "latest"
    opts["logroot"] = "logdir"
    model, data_info, _ = Trainer.construct_test_model(opts, return_refs=False, force_reload=False)
    field = model.fields.field_params["fg"]
    art = field.warp.articulation
    print("fg field %s, warp %s, articulation %s" % (type(field).__name__, type(field.warp).__name__,
                                                    type(art).__name__), flush=True)
    # this branch's fields work in an 'ideal' space; metric = ideal * scale_const / exp(logscale) (NeRF.forward)
    scale = float(field.logscale.exp()) / float(getattr(field, "scale_const", 1.0))
    if hasattr(field, "symm_ratio"):
        field.symm_ratio = 0.0  # no random mirroring while querying colours

    # rest mesh (canonical), decimated
    mesh = field.extract_canonical_mesh(grid_size=200, level=0.0, inst_id=None, vis_thresh=0.0, use_extend_aabb=False)
    full, small = os.path.join(out_dir, "a2s_cat_full.ply"), os.path.join(out_dir, "a2s_cat.obj")
    mesh.export(full)
    code = ("import open3d as o3d,sys; m=o3d.io.read_triangle_mesh(sys.argv[1]);"
            "m=m.simplify_quadric_decimation(int(sys.argv[3])); m.remove_unreferenced_vertices();"
            "m.remove_degenerate_triangles(); o3d.io.write_triangle_mesh(sys.argv[2], m)")
    subprocess.run([os.environ["DECIMATE_PY"], "-c", code, full, small, str(faces)], check=True)
    mesh = trimesh.load(small, process=False)
    tex_size = int(os.environ.get("CAT_TEX", "0"))
    bake = None
    if tex_size:
        from cat_uv_bake import fill_and_save, uv_bake
        bake = uv_bake(small, tex_size)
        mesh = trimesh.Trimesh(vertices=bake["pos"], faces=bake["faces"], process=False)
    V = len(mesh.vertices)
    print("rest mesh: %d vertices, %d faces; object scale %.4f" % (V, len(mesh.faces), scale), flush=True)
    xyz = torch.tensor(mesh.vertices, dtype=torch.float32, device=dev)

    # colours: the appearance code (average, or one of 16 frames spread over the month) whose coat lightness is
    # closest to the real cat's (CIELAB L about 74, measured on real frames in the C1 colour test)
    emb = getattr(field, "appr_embedding", None)
    total = int(data_info["frame_info"]["frame_offset_raw"][-1])

    def colours(fid=None, pts=None):
        pts = xyz if pts is None else pts
        orig = emb.get_vals if emb is not None else None
        if emb is not None and fid is None:
            emb.get_vals = lambda frame_id=None: emb.get_mean_vals().reshape(1, -1)
        out = []
        for k in range(0, len(pts), 200000):
            p = pts[k:k + 200000]
            rgb = field.forward(p[None, :, None], dir=torch.zeros_like(p)[None, :, None],
                                frame_id=torch.tensor([fid or 0], dtype=torch.long, device=dev), inst_id=None)[0]
            out.append(rgb[0, :, 0].clamp(0, 1))
        if emb is not None:
            emb.get_vals = orig
        return torch.cat(out)

    def lightness(c):
        lin = torch.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
        y = lin @ torch.tensor([0.2126729, 0.7151522, 0.0721750], device=c.device)
        f = torch.where(y > (6 / 29) ** 3, y.clamp(min=1e-8) ** (1 / 3), y / (3 * (6 / 29) ** 2) + 4 / 29)
        return float((116 * f - 16).mean())

    cands = [None] + [int(x) for x in np.linspace(0, total - 1, 16).round()]
    scored = [(abs(lightness(colours(c)) - 74.0), c) for c in cands]
    best = min(scored, key=lambda x: x[0])[1]
    print("colour code: %s (lightness off by %.1f; average code off by %.1f)" % (
        "average" if best is None else "frame %d" % best, min(scored)[0], scored[0][0]), flush=True)
    # then match exposure: one gain on linear light so the coat's average lightness is the real cat's
    c = colours(best)
    lin = torch.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    to_srgb = lambda x: torch.where(x <= 0.0031308, 12.92 * x, 1.055 * x.clamp(min=1e-8) ** (1 / 2.4) - 0.055).clamp(0, 1)
    lo_g, hi_g = 0.5, 4.0
    for _ in range(30):
        g = (lo_g + hi_g) / 2
        lo_g, hi_g = (g, hi_g) if lightness(to_srgb(lin * g)) < 74.0 else (lo_g, g)
    print("exposure gain %.3f -> lightness %.1f" % (g, lightness(to_srgb(lin * g))), flush=True)
    col = (to_srgb(lin * g) * 255).round().byte().cpu().numpy()
    if bake is not None:  # the same code and exposure, at every texel
        tp = torch.tensor(bake["texel_pos"], dtype=torch.float32, device=dev)
        ct = colours(best, tp)
        ct = to_srgb(torch.where(ct <= 0.04045, ct / 12.92, ((ct + 0.055) / 1.055) ** 2.4) * g)
        fill_and_save(ct.cpu().numpy(), bake["texel_rc"], tex_size, os.path.join(out_dir, "a2s_cat_tex.jpg"))
        print("texture: %d texels baked into a2s_cat_tex.jpg" % len(tp), flush=True)

    # skinning weights at the rest pose (as SkinningWarp.forward computes them for forward warping)
    rest = art.get_mean_vals()  # ((1,B,4),(1,B,4))
    B = rest[0].shape[-2]
    rest_x = (rest[0][:, None, None].expand(1, 1, V, B, 4), rest[1][:, None, None].expand(1, 1, V, B, 4))
    skin, _ = field.warp.skinning_model(xyz[None, None], rest_x, None, None)
    prob = skin.softmax(-1)[0, 0]  # (V,B)
    w4, i4 = prob.topk(4, -1)
    w4 = w4 / w4.sum(-1, keepdim=True)
    print("bones %d; top-4 weights cover %.4f of the weight on average" % (B, float(prob.topk(4, -1)[0].sum(-1).mean())),
          flush=True)

    with open(pkl_path, "rb") as f:
        d = pickle.load(f)

    def bone_motion(so3):
        """(M,B,3) joint angles -> (M,B,3,4) rest->pose transforms in metres"""
        so3 = torch.as_tensor(so3, dtype=torch.float32, device=dev)
        t_art = art.forward(art.time_embedding(torch.zeros(len(so3), dtype=torch.long, device=dev)), None, override_so3=so3,
                            override_local_rest_joints=art.compute_rel_rest_joints().repeat(len(so3), 1, 1))
        r = (rest[0].expand_as(t_art[0]), rest[1].expand_as(t_art[1]))
        se3 = dual_quaternion_to_se3(dual_quaternion_mul(t_art, dual_quaternion_inverse(r)))  # (M,B,4,4)
        se3[..., :3, 3] *= metric
        return se3[..., :3, :]

    # metres per model unit: fit the model's joint positions to the recorded ones (joints3D, the cat's own frame)
    so3_0 = torch.as_tensor(np.asarray(d["poses"][0]), dtype=torch.float32, device=dev)
    t_art0 = art.forward(art.time_embedding(torch.zeros(len(so3_0), dtype=torch.long, device=dev)), None,
                         override_so3=so3_0, override_local_rest_joints=art.compute_rel_rest_joints().repeat(len(so3_0), 1, 1))
    ja = dual_quaternion_to_se3(t_art0)[..., :3, 3].reshape(-1, 3).cpu().numpy()
    jb = np.asarray(d["joints3D"][0], dtype=np.float64).reshape(-1, 3)
    metric = float((ja * jb).sum() / (ja * ja).sum())
    print("metres per model unit %.4f (1/exp(logscale)*scale_const would be %.4f); joint fit residual median %.1f mm, "
          "max %.1f mm" % (metric, 1 / scale, 1000 * np.median(np.linalg.norm(metric * ja - jb, axis=1)),
                           1000 * np.max(np.linalg.norm(metric * ja - jb, axis=1))), flush=True)

    # recorded joint angles, video by video, in the order of a2s_room_export.py
    n = len(d["se3"])
    starts = [0] + [i + 1 for i in range(n - 1) if not same_video(d, i)]
    so3_all = []
    for s, e in zip(starts, starts[1:] + [n]):
        so3_all.append(np.concatenate([np.asarray(d["poses"][s])] + [np.asarray(d["poses"][i])[-1:] for i in range(s + 1, e)]))
    so3_all = np.concatenate(so3_all)  # (F,J,3)
    print("recorded frames %d, joint angles %s" % (len(so3_all), so3_all.shape[1:]), flush=True)
    bones = np.concatenate([bone_motion(so3_all[k:k + 2048]).cpu().numpy() for k in range(0, len(so3_all), 2048)])

    # self-check against Lab4D's own warp on a few frames
    for k in (0, len(so3_all) // 2, len(so3_all) - 1):
        so3 = torch.as_tensor(so3_all[k:k + 1], dtype=torch.float32, device=dev)
        t_art = art.forward(art.time_embedding(torch.zeros(1, dtype=torch.long, device=dev)), None, override_so3=so3,
                            override_local_rest_joints=art.compute_rel_rest_joints())
        ref = field.warp(xyz[None, None], None, None, samples_dict={"t_articulation": t_art, "rest_articulation": rest})
        ref = ref[0, 0] * metric
        T = torch.as_tensor(bones[k], device=dev)  # (B,3,4)
        pts = (T[i4] @ torch.cat([xyz * metric, torch.ones(V, 1, device=dev)], -1)[:, None, :, None])[..., 0]  # (V,4,3)
        lbs = (pts * w4[..., None]).sum(1)
        err = (lbs - ref).norm(dim=-1)
        print("frame %d: 4-bone blend vs Lab4D warp, median %.2f mm, max %.1f mm" % (
            k, 1000 * float(err.median()), 1000 * float(err.max())), flush=True)

    # joints3D from the data vs the mesh, to confirm the units match
    j3 = np.asarray(d["joints3D"][0][0])
    print("recorded joints3D extent (m): %s; rest mesh extent (m): %s" % (
        np.ptp(j3, 0).round(3).tolist(), np.ptp(mesh.vertices * metric, 0).round(3).tolist()), flush=True)

    blobs = [("faces", np.asarray(mesh.faces, dtype=np.uint32)), ("pos", (mesh.vertices * metric).astype(np.float32)),
             ("col", col), ("skin_idx", i4.byte().cpu().numpy()), ("skin_w", w4.float().cpu().numpy()),
             *([("uv", bake["uv"].astype(np.float32))] if bake is not None else []),
             ("bones", bones.reshape(len(bones), B, 12).astype(np.float16))]
    meta, size = pack(os.path.join(out_dir, "a2s_cat.bin"), blobs)
    with open(os.path.join(out_dir, "a2s_cat.json"), "w") as fh:
        json.dump({"blobs": meta, "bytes": size, "vertices": V, "bones": B, "frames": len(bones),
                   "units": "metres, the cat's own (root) frame",
                   **({"texture": "a2s_cat_tex.jpg"} if bake is not None else {})}, fh)
    print("a2s_cat: %d bytes" % size, flush=True)

    if rollouts:
        ro = json.load(open(rollouts))
        walk_bones = []
        for run in ro["runs"]:
            so3 = np.concatenate([np.asarray(run["past_joints"]), np.asarray(run["joints"])])
            walk_bones.append(bone_motion(so3).cpu().numpy().reshape(len(so3), B, 12).astype(np.float16))
        meta, size = pack(os.path.join(out_dir, "a2s_walks.bin"), [("bones", np.concatenate(walk_bones))])
        with open(os.path.join(out_dir, "a2s_walks.json"), "w") as fh:
            json.dump({"blobs": meta, "bytes": size, "frames_per_run": [len(w) for w in walk_bones]}, fh)
        print("a2s_walks: %d runs, %d bytes" % (len(walk_bones), size), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], *(int(x) for x in sys.argv[4:5]), *sys.argv[5:6])
