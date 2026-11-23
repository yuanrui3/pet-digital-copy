"""Real cats' gait from video (work/pets/PROTOCOL_CAT.md, "Real cats' gait from video"), step 1: 2D keypoints on every
raw frame (30 a second) of the cat-85 clips with SuperAnimal-Quadruped (HRNet-w32; Ye et al.; academic, non-commercial
licence), run here without DeepLabCut: timm's hrnet_w32 (features only, the stride-4 branch) and the checkpoint's
1x1 transposed-convolution heatmap head, ImageNet normalisation, top-down on a square crop of the cat.
The crop comes from the set's own outlines (crop-256 masks of the kept frames), interpolated over the raw frames
between them, enlarged by MARGIN. Keypoint = heatmap peak with a quadratic sub-pixel step; score = the peak value
clipped to [0, 1]. GPU.
usage: python sa_pose.py MODEL_PTH REAL_DIR RAW_DIR OUT_DIR [clip ...]
  REAL_DIR: experiments_judge/real (Annotations, kept_frames.json); RAW_DIR: .../JPEGImagesRaw/Full-Resolution
Writes OUT_DIR/<clip>.npz (kp: frames x 39 x 3 in raw pixels with score; raw: raw frame numbers; box) and, with SHEET
set, sheet.jpg (20 random frames with the legs drawn)."""
import json
import os
import sys

import numpy as np
import timm
import torch
import torch.nn.functional as Fn
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sa_load import load  # noqa: E402

NAMES = ["nose", "upper_jaw", "lower_jaw", "mouth_end_right", "mouth_end_left", "right_eye", "right_earbase", "right_earend",
         "right_antler_base", "right_antler_end", "left_eye", "left_earbase", "left_earend", "left_antler_base",
         "left_antler_end", "neck_base", "neck_end", "throat_base", "throat_end", "back_base", "back_end", "back_middle",
         "tail_base", "tail_end", "front_left_thai", "front_left_knee", "front_left_paw", "front_right_thai",
         "front_right_knee", "front_right_paw", "back_left_paw", "back_left_thai", "back_right_thai", "back_left_knee",
         "back_right_knee", "back_right_paw", "belly_bottom", "body_middle_right", "body_middle_left"]
SIZE, MARGIN, BATCH = 256, 1.25, 32
MEAN, STD = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])


def build(path, dev):
    sd = load(path)["model_state_dict"]
    net = timm.create_model("hrnet_w32", pretrained=False, features_only=True, feature_location="", out_indices=(1, 2, 3, 4))
    bb = {k[len("backbone.model."):]: v for k, v in sd.items() if k.startswith("backbone.model.")}
    missing, unexpected = net.load_state_dict(bb, strict=False)
    assert not missing, missing[:5]
    W, b = sd["heads.bodypart.heatmap_head.model.weight"], sd["heads.bodypart.heatmap_head.model.bias"]
    return net.to(dev).eval(), W.to(dev), b.to(dev), unexpected


@torch.no_grad()
def predict(net, W, b, crops, dev):
    x = torch.tensor(((np.stack(crops) / 255.0 - MEAN) / STD).transpose(0, 3, 1, 2), dtype=torch.float32, device=dev)
    f = net(x)[0]
    hm = Fn.conv_transpose2d(f, W, b).float().cpu().numpy()  # (n, 39, 64, 64)
    n, K, H, Wd = hm.shape
    flat = hm.reshape(n, K, -1)
    i = flat.argmax(-1)
    iy, ix = i // Wd, i % Wd
    out = np.zeros((n, K, 3))
    for a in range(n):
        for k in range(K):
            y, xx = iy[a, k], ix[a, k]
            dx = dy = 0.0
            if 0 < xx < Wd - 1:
                l, c, r = hm[a, k, y, xx - 1], hm[a, k, y, xx], hm[a, k, y, xx + 1]
                den = l - 2 * c + r
                dx = 0.5 * (l - r) / den if den < 0 else 0.0
            if 0 < y < H - 1:
                u, c, d = hm[a, k, y - 1, xx], hm[a, k, y, xx], hm[a, k, y + 1, xx]
                den = u - 2 * c + d
                dy = 0.5 * (u - d) / den if den < 0 else 0.0
            out[a, k] = ((xx + dx + 0.5) * 4 - 0.5, (y + dy + 0.5) * 4 - 0.5, np.clip(hm[a, k, y, xx], 0, 1))
    return out


def boxes(ann, kept, raws):
    """a square box per raw frame (raw pixels), from the kept frames' outlines, interpolated"""
    M, K = np.load(os.path.join(ann, "crop-256.npy")), np.load(os.path.join(ann, "crop-256-crop2raw.npy"))
    cx, cy, sz, ok = [], [], [], []
    for k in range(len(M)):
        m = M[k, ..., 0] if M.ndim == 4 else M[k]
        r, c = np.nonzero(m)
        if len(r) < 30:
            continue
        x0, x1 = K[k, 0] * c.min() + K[k, 2], K[k, 0] * (c.max() + 1) + K[k, 2]
        y0, y1 = K[k, 1] * r.min() + K[k, 3], K[k, 1] * (r.max() + 1) + K[k, 3]
        cx.append((x0 + x1) / 2)
        cy.append((y0 + y1) / 2)
        sz.append(max(x1 - x0, y1 - y0))
        ok.append(kept[k])
    ok = np.array(ok, float)
    return np.stack([np.interp(raws, ok, cx), np.interp(raws, ok, cy), np.interp(raws, ok, sz) * MARGIN], 1)


def crop(img, box):
    cx, cy, s = box
    x0, y0 = cx - s / 2, cy - s / 2
    im = img.transform((SIZE, SIZE), Image.EXTENT, (x0, y0, x0 + s, y0 + s), resample=Image.BILINEAR)
    return np.asarray(im, dtype=np.float32)


def main():
    model, real, raw_dir, out = sys.argv[1:5]
    clips = sys.argv[5:]
    os.makedirs(out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net, W, b, unexpected = build(model, dev)
    print("model loaded; unused checkpoint layers (classification only): %d" % len(unexpected), flush=True)
    kept_all = json.load(open(os.path.join(real, "kept_frames.json")))
    ann_base = os.path.join(real, "database", "processed", "Annotations", "Full-Resolution")
    sheet = []
    rng = np.random.default_rng(0)
    for clip in clips:
        folder = os.path.join(raw_dir, clip)
        raws = sorted(int(f[:5]) for f in os.listdir(folder) if f.endswith(".jpg"))
        bx = boxes(os.path.join(ann_base, clip), kept_all[clip], np.array(raws, float))
        kp = np.zeros((len(raws), len(NAMES), 3))
        for s in range(0, len(raws), BATCH):
            ims = [Image.open(os.path.join(folder, "%05d.jpg" % r)).convert("RGB") for r in raws[s:s + BATCH]]
            p = predict(net, W, b, [crop(im, bx[s + j]) for j, im in enumerate(ims)], dev)
            sc = bx[s:s + len(ims), 2:3] / SIZE
            kp[s:s + len(ims), :, 0] = bx[s:s + len(ims), 0:1] - bx[s:s + len(ims), 2:3] / 2 + p[:, :, 0] * sc
            kp[s:s + len(ims), :, 1] = bx[s:s + len(ims), 1:2] - bx[s:s + len(ims), 2:3] / 2 + p[:, :, 1] * sc
            kp[s:s + len(ims), :, 2] = p[:, :, 2]
        np.savez(os.path.join(out, "%s.npz" % clip), kp=kp.astype(np.float32), raw=np.array(raws), box=bx.astype(np.float32))
        legs = [NAMES.index(n) for n in NAMES if "paw" in n]
        print("%s: %d raw frames; paw score median %.2f" % (clip, len(raws), float(np.median(kp[:, legs, 2]))), flush=True)
        if os.environ.get("SHEET"):
            for j in rng.choice(len(raws), 2, replace=False):
                sheet.append((clip, raws[j], kp[j], bx[j]))
    if sheet:
        tiles = []
        col = {"front_left": (255, 60, 60), "front_right": (255, 170, 0), "back_left": (60, 160, 255), "back_right": (60, 230, 120)}
        for clip, r, k, box in sheet[:20]:
            im = Image.open(os.path.join(raw_dir, clip, "%05d.jpg" % r)).convert("RGB")
            cx, cy, s = box
            x0, y0 = cx - s / 2, cy - s / 2
            d = ImageDraw.Draw(im)
            for leg, c in col.items():
                pts = [k[NAMES.index(leg + "_" + j)] for j in ("thai", "knee", "paw")]
                d.line([(p[0], p[1]) for p in pts], fill=c, width=max(2, int(s / 120)))
                for p in pts:
                    rr = max(3, int(s / 80))
                    d.ellipse((p[0] - rr, p[1] - rr, p[0] + rr, p[1] + rr), outline=c, width=2)
            for j in ("neck_base", "back_middle", "tail_base"):
                p = k[NAMES.index(j)]
                d.ellipse((p[0] - 4, p[1] - 4, p[0] + 4, p[1] + 4), fill=(255, 255, 255))
            t = im.crop((int(x0), int(y0), int(x0 + s), int(y0 + s))).resize((300, 300))
            ImageDraw.Draw(t).text((4, 4), "%s %d" % (clip[-4:], r), fill=(255, 255, 0))
            tiles.append(np.asarray(t))
        while len(tiles) % 5:
            tiles.append(np.zeros_like(tiles[0]))
        rows = [np.concatenate(tiles[i:i + 5], 1) for i in range(0, len(tiles), 5)]
        Image.fromarray(np.concatenate(rows, 0)).save(os.path.join(out, "sheet.jpg"), quality=85)


if __name__ == "__main__":
    main()
