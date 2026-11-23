"""The video-model judge (work/pets/PROTOCOL_CAT.md, "A video-model judge"): V-JEPA 2 features of silhouette clips.

V-JEPA 2 ViT-L (facebook/vjepa2-vitl-fpc64-256, MIT), encoder only: each 16-frame 256 px clip (white on black, the
same in all three colour channels, ImageNet normalisation as the model's processor) -> the encoder's last tokens
averaged to one 1024-number vector; the clip and its mirror image are both run and their vectors averaged, so the
direction the cat walks across the picture does not matter. GPU. env-judge.
usage: python jepa_feat.py MODEL_DIR CLIPS_NPZ [CLIPS_NPZ ...]
Writes next to each CLIPS_NPZ: <name>_feat.npy (clips x 1024, float32) and <name>_feat.json (timing)."""
import json
import os
import sys
import time

import numpy as np
import torch
from transformers import AutoModel

model_dir, paths = sys.argv[1], sys.argv[2:]
BATCH = int(os.environ.get("BATCH", 8))
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)
t_load = time.time()
model = AutoModel.from_pretrained(model_dir).cuda().eval()
print("model loaded in %.0f s" % (time.time() - t_load), flush=True)
mean, std = MEAN.cuda(), STD.cuda()


@torch.inference_mode()
def encode(x):
    """x (B, 16, 256, 256) uint8 -> (B, 1024)"""
    v = torch.from_numpy(x).cuda().float().div_(255.0)
    v = torch.cat([v, v.flip(-1)])                       # the clips and their mirror images
    v = (v[:, :, None].expand(-1, -1, 3, -1, -1) - mean) / std
    with torch.autocast("cuda", dtype=torch.bfloat16):
        try:
            h = model(pixel_values_videos=v, skip_predictor=True).last_hidden_state
        except TypeError:
            h = model.get_vision_features(pixel_values_videos=v)
    f = h.float().mean(1)
    n = len(x)
    return ((f[:n] + f[n:]) / 2).cpu().numpy()


for p in paths:
    t0 = time.time()
    clips = np.load(p)["clips"]
    torch.cuda.synchronize()
    t1 = time.time()
    feats = np.concatenate([encode(clips[i:i + BATCH]) for i in range(0, len(clips), BATCH)]) if len(clips) else np.zeros((0, 1024))
    torch.cuda.synchronize()
    t2 = time.time()
    np.save(p.replace(".npz", "_feat.npy"), feats.astype(np.float32))
    json.dump({"clips": int(len(clips)), "load_s": round(t1 - t0, 1), "gpu_s": round(t2 - t1, 1),
               "norm_median": float(np.median(np.linalg.norm(feats, axis=1))) if len(feats) else None},
              open(p.replace(".npz", "_feat.json"), "w"))
    print("%s: %d clips, %.1f s on the GPU" % (os.path.basename(p), len(clips), t2 - t1), flush=True)
