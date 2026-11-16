"""Paints the real video frames onto the C1 cat's UV-mapped surface, for fur detail in the web viewer
(demo, not a test).

Run from the Lab4D code folder (env-lab4d):
  UV_NPZ=.../c1_uv.npz NET_TEX=.../cat_tex_fixed.jpg python $D/tools/cat_photo_texture.py \
      --flagfile=logdir/LOG/opts.log --load_suffix latest --cat_out OUTDIR

For every frame of the video: pose the mesh with the model's own camera, body motion and lens; find the texels
the camera sees (depth test against the other texels, inside the cat's mask, facing the camera); read the photo's
colour there. Each texel then mixes the frames that see it most head-on. Texels that no frame sees well keep the
colour network's colour (NET_TEX, same UV layout). All 76 frames are used, including the 19 whose colours the C1
test held back from the model: the test is over, and this texture is a copy of the photos, not a model output.

Writes OUTDIR/cat_tex_photo_{best_raw,best,soft,detail}.jpg (ways of mixing; see below), photo_cover.jpg (how head-on each texel
was seen) and photo.json (per-frame checks: how well the posed mesh's outline matches the cat's mask).
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cat_eval  # noqa: E402,F401  (Lab4D folder on sys.path, --cat_out)
from cat_uv_bake_fill import fill_and_save  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from absl import app  # noqa: E402

from lab4d.config import get_config  # noqa: E402
from lab4d.dataloader import data_utils  # noqa: E402
from lab4d.engine.model import dvr_model  # noqa: E402
from lab4d.engine.trainer import Trainer  # noqa: E402



def texel_faces(uv, faces, size):
    """Face id and barycentric weights of every texel centre that falls inside a UV triangle.
    Rows follow three.js: row 0 holds v = 1. Texel (r, c) has its centre at u = (c + 0.5) / size, v = 1 - (r + 0.5) / size."""
    fid = np.full((size, size), -1, np.int64)
    bary = np.zeros((size, size, 3), np.float32)
    P = np.stack([uv[:, 0] * size - 0.5, (1 - uv[:, 1]) * size - 0.5], 1)  # (col, row), texel centres at integers
    for k, (a, b, c) in enumerate(faces):
        pa, pb, pc = P[a], P[b], P[c]
        d = (pb[1] - pc[1]) * (pa[0] - pc[0]) + (pc[0] - pb[0]) * (pa[1] - pc[1])
        if abs(d) < 1e-12:
            continue
        lo = np.clip(np.floor(np.minimum(np.minimum(pa, pb), pc)).astype(int), 0, size - 1)
        hi = np.clip(np.ceil(np.maximum(np.maximum(pa, pb), pc)).astype(int), 0, size - 1)
        cs, rs = np.meshgrid(np.arange(lo[0], hi[0] + 1), np.arange(lo[1], hi[1] + 1))
        x, y = cs.ravel().astype(np.float64), rs.ravel().astype(np.float64)
        w0 = ((pb[1] - pc[1]) * (x - pc[0]) + (pc[0] - pb[0]) * (y - pc[1])) / d
        w1 = ((pc[1] - pa[1]) * (x - pc[0]) + (pa[0] - pc[0]) * (y - pc[1])) / d
        w2 = 1 - w0 - w1
        ins = (w0 >= -1e-6) & (w1 >= -1e-6) & (w2 >= -1e-6)
        r, cc = rs.ravel()[ins], cs.ravel()[ins]
        fid[r, cc] = k
        bary[r, cc] = np.stack([w0[ins], w1[ins], w2[ins]], 1)
    rc = np.argwhere(fid >= 0)
    return rc, fid[rc[:, 0], rc[:, 1]], bary[rc[:, 0], rc[:, 1]]


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
    mapping = [int(x) for x in np.asarray(data_info["frame_info"]["frame_mapping"])]  # raw ids = file names
    seq = opts["seqname"]
    inst = torch.tensor([0], device=dev)
    # the video's own frame and mask folders, as the sequence's config names them
    import configparser
    cfg = configparser.RawConfigParser()
    cfg.read("database/configs/%s.config" % seq)
    img_dir = cfg.get("data_0", "img_path").rstrip("/")
    frame_path, mask_path = img_dir + "/%05d.jpg", img_dir.replace("JPEGImages", "Annotations") + "/%05d.npy"

    uvd = np.load(os.environ["UV_NPZ"])
    size = int(uvd["size"])
    faces_np = uvd["faces"].astype(np.int64)
    rc, tf, tb = texel_faces(uvd["uv"].astype(np.float64), faces_np, size)
    verts = torch.tensor(uvd["pos"], dtype=torch.float32, device=dev)
    faces = torch.tensor(faces_np, device=dev)
    tf_t = torch.tensor(tf, device=dev)
    tb_t = torch.tensor(tb, device=dev)
    tri = faces[tf_t]  # (T,3) vertex ids of each texel's triangle
    T = len(rc)

    # check: my texel positions against open3d's (same UV layout, so they should agree to a fraction of a texel)
    tpos = (verts[tri] * tb_t[..., None]).sum(1).cpu().numpy()
    idx = np.full((size, size), -1, np.int64)
    idx[uvd["texel_rc"][:, 0], uvd["texel_rc"][:, 1]] = np.arange(len(uvd["texel_rc"]))
    j = idx[rc[:, 0], rc[:, 1]]
    ok = j >= 0
    diag = float((verts.max(0).values - verts.min(0).values).norm())
    pos_err = float(np.median(np.linalg.norm(tpos[ok] - uvd["texel_pos"][j[ok]], axis=1))) / diag
    print("texels: %d (open3d had %d; %.1f%% shared), position vs open3d: median %.5f of the cat's size" % (
        T, len(uvd["texel_rc"]), 100 * ok.mean(), pos_err), flush=True)

    # outward normals: the sign that makes face normals point away from the middle on average
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    fn0 = torch.cross(v1 - v0, v2 - v0, dim=1)
    sign = 1.0 if float((fn0 * ((v0 + v1 + v2) / 3 - verts.mean(0))).sum()) > 0 else -1.0
    eps = 0.015 * diag  # depth tolerance for "not hidden"

    nF = len(mapping)
    Wt = torch.zeros((nF, T), dtype=torch.float16, device=dev)
    Cl = torch.zeros((nF, T, 3), dtype=torch.uint8, device=dev)
    checks = []
    for i, raw in enumerate(mapping):
        fid = torch.tensor([raw], dtype=torch.long, device=dev)
        quat, trans = field.camera_mlp.get_vals(fid)
        sd = {}
        sd["t_articulation"], sd["rest_articulation"] = field.warp.articulation.get_vals_and_mean(fid)
        xc = field.forward_warp(verts[None, None], (quat, trans), fid, inst, samples_dict=sd)[0, 0].float()
        fx, fy, px, py = [float(x) for x in model.intrinsics.get_vals(fid)[0]]
        fn = torch.cross(xc[faces[:, 1]] - xc[faces[:, 0]], xc[faces[:, 2]] - xc[faces[:, 0]], dim=1) * sign
        vn = torch.zeros_like(xc)
        for k in range(3):
            vn.index_add_(0, faces[:, k], fn)
        vn = F.normalize(vn, dim=1)
        p = (xc[tri] * tb_t[..., None]).sum(1)
        n = F.normalize((vn[tri] * tb_t[..., None]).sum(1), dim=1)
        z = p[:, 2]
        u = fx * p[:, 0] / z + px
        v = fy * p[:, 1] / z + py
        cosv = -(n * F.normalize(p, dim=1)).sum(1)

        img = cv2.imread(frame_path % raw)[:, :, ::-1].copy()
        H, Wd = img.shape[:2]
        m = np.load(mask_path % raw) > 0
        me = cv2.erode(m.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0  # 4 px in from the outline
        mt = torch.tensor(m, device=dev).view(-1)
        met = torch.tensor(me, device=dev).view(-1)

        ui, vi = u.round().long(), v.round().long()
        inb = (ui >= 0) & (ui < Wd) & (vi >= 0) & (vi < H) & (z > 0)
        pix = (vi * Wd + ui).clamp(0, H * Wd - 1)
        zb = torch.full((H * Wd,), float("inf"), device=dev)
        zb.scatter_reduce_(0, pix[inb], z[inb], reduce="amin")
        zb = -F.max_pool2d(-zb.view(1, 1, H, Wd), 5, 1, 2).view(-1)  # nearest surface within 2 px
        vis = inb & (z <= zb[pix] + eps)
        use = vis & met[pix] & (cosv > 0.1)
        Wt[i] = torch.where(use, cosv, torch.zeros_like(cosv)).half()
        it = torch.tensor(img, device=dev).permute(2, 0, 1)[None].float()
        grid = torch.stack([u / (Wd - 1) * 2 - 1, v / (H - 1) * 2 - 1], -1).view(1, 1, T, 2)
        Cl[i] = F.grid_sample(it, grid, align_corners=True)[0, :, 0].T.round().clamp(0, 255).byte()

        # checks: outline of the posed mesh against the mask, and how many facing, unhidden texels land on the cat
        sil = torch.isfinite(zb)
        iou = float((sil & mt).sum() / (sil | mt).sum())
        facing = vis & (cosv > 0.1)
        on_cat = float((facing & mt[pix]).sum() / facing.sum().clamp(min=1))
        checks.append({"raw": raw, "iou_outline_vs_mask": round(iou, 4), "facing_texels_on_cat": round(on_cat, 4),
                       "texels_used": int(use.sum())})
        if i == 0:
            print("frame %d: focal %.1f, centre (%.1f, %.1f), image %dx%d" % (raw, fx, px, py, Wd, H), flush=True)
    ious = [c["iou_outline_vs_mask"] for c in checks]
    oncat = [c["facing_texels_on_cat"] for c in checks]
    print("outline vs mask IoU: median %.3f, min %.3f; facing texels on the cat: median %.3f, min %.3f" % (
        np.median(ious), np.min(ious), np.median(oncat), np.min(oncat)), flush=True)

    # mix the frames.
    # 1. exposure: each frame's colours divided by its own gain against the colour network's coat (median ratio
    #    over the texels it sees well), times the median gain, so all frames share one exposure.
    net = cv2.imread(os.environ["NET_TEX"])[:, :, ::-1].astype(np.float32) / 255
    net_t = torch.tensor(net[rc[:, 0], rc[:, 1]], device=dev)
    gains = []
    for i in range(nF):
        sel = (Wt[i] > 0.5) & (net_t.min(1).values > 0.05)
        ratio = Cl[i, sel].float() / 255 / net_t[sel]
        gains.append(ratio.median(0).values if sel.sum() > 1000 else torch.ones(3, device=dev))
    gains = torch.stack(gains)  # (nF,3)
    G = gains.median(0).values
    corr = G[None] / gains
    print("exposure gains per frame: %.2f..%.2f (median %s)" % (float(gains.mean(1).min()), float(gains.mean(1).max()),
                                                                [round(float(x), 3) for x in G]), flush=True)
    # 2. agreement: a frame counts at a texel only as far as its colour agrees with the median of the frames that
    #    see that texel well (rejects frames whose pose slid an eye onto the cheek); 3. outputs:
    #    best = the most head-on agreeing frame (sharp, patchy); soft = agreeing frames mixed (smooth);
    #    detail = soft's broad colour plus best's fine detail (made below, in the image).
    outs = {"best_raw": [], "best": [], "soft": [], "best_c": [], "best_loose": []}
    wmax_all = []
    for s in range(0, T, 300000):
        w = Wt[:, s:s + 300000].float()
        c = Cl[:, s:s + 300000].float() / 255
        fb = (net_t[s:s + 300000] * G).clamp(0, 1)
        wmax, best0 = w.max(0)
        ar = torch.arange(w.shape[1], device=dev)
        alpha = ((wmax - 0.2) / 0.3).clamp(0, 1)[:, None]  # seen within 60 degrees of head-on -> photo only
        outs["best_raw"].append(alpha * c[best0, ar] + (1 - alpha) * net_t[s:s + 300000])
        c = (c * corr[:, None, :]).clamp(0, 1)
        good = w > 0.3
        med = torch.where(good[..., None], c, torch.full_like(c, float("nan"))).nanmedian(0).values
        ref = torch.where((good.sum(0) >= 3)[:, None], med, fb)
        agree = torch.exp(-(((c - ref[None]).norm(dim=-1) / 0.1) ** 2))
        sc = (w ** 4) * agree * (w > 0)
        smax, bi = sc.max(0)
        best = torch.where((smax > 1e-6)[:, None], c[bi, ar], ref)
        soft = (sc[..., None] * c).sum(0) / sc.sum(0).clamp(min=1e-8)[:, None]
        soft = torch.where((sc.sum(0) > 1e-6)[:, None], soft, ref)
        outs["best"].append(alpha * best + (1 - alpha) * fb)
        outs["soft"].append(alpha * soft + (1 - alpha) * fb)
        outs["best_c"].append(alpha * c[best0, ar] + (1 - alpha) * fb)  # most head-on frame, exposure evened out
        sl = (w ** 4) * torch.exp(-(((c - ref[None]).norm(dim=-1) / 0.25) ** 2)) * (w > 0)  # rejects only clear outliers
        sm, bl = sl.max(0)
        outs["best_loose"].append(alpha * torch.where((sm > 1e-6)[:, None], c[bl, ar], ref) + (1 - alpha) * fb)
        wmax_all.append(wmax)
    outs = {k: torch.cat(v).cpu().numpy() for k, v in outs.items()}

    def lowpass(vals, sigma=6.0):  # blur within the charts only (texels outside them carry no weight)
        img = np.zeros((size, size, 3), np.float32)
        m = np.zeros((size, size), np.float32)
        img[rc[:, 0], rc[:, 1]] = vals
        m[rc[:, 0], rc[:, 1]] = 1
        bi_ = cv2.GaussianBlur(img, (0, 0), sigma)
        bm = cv2.GaussianBlur(m, (0, 0), sigma)
        return (bi_ / np.maximum(bm, 1e-6)[..., None])[rc[:, 0], rc[:, 1]]

    base = lowpass(outs["soft"])
    outs["detail"] = np.clip(base + outs["best"] - lowpass(outs["best"]), 0, 1)
    outs["detail_raw"] = np.clip(base + outs["best_c"] - lowpass(outs["best_c"]), 0, 1)
    outs["detail_loose"] = np.clip(base + outs["best_loose"] - lowpass(outs["best_loose"]), 0, 1)
    for name, vals in outs.items():
        fill_and_save(vals, rc, size, os.path.join(out_dir, "cat_tex_photo_%s.jpg" % name))
    wmax = torch.cat(wmax_all).cpu().numpy()
    cover = np.zeros((size, size), np.uint8)
    cover[rc[:, 0], rc[:, 1]] = (np.clip(wmax, 0, 1) * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(out_dir, "photo_cover.jpg"), cover)
    summary = {"frames": nF, "texels": T, "size": size, "texel_position_vs_open3d": pos_err,
               "share_seen_head_on_within_60deg": float((wmax >= 0.5).mean()),
               "share_seen_at_all": float((wmax > 0.1).mean()),
               "share_network_colour_only": float((wmax <= 0.2).mean()),
               "iou_median": float(np.median(ious)), "iou_min": float(np.min(ious)),
               "facing_on_cat_median": float(np.median(oncat)), "per_frame": checks}
    with open(os.path.join(out_dir, "photo.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    print({k: v for k, v in summary.items() if k != "per_frame"}, flush=True)


if __name__ == "__main__":
    app.run(main)
