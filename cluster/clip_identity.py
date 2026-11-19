"""Screening the close-up clips for a rebuild (work/pets/PROTOCOL_CAT.md, "Better data: screening the close-up
clips"): which cat-85 clips show this cat? The cat test's identity check (cat_identity.py: MegaDescriptor-L-384; the
template is the mean of the Body tab clip's real frames; the threshold is the 99th percentile of the Oxford-IIIT Pet
cats' similarity to it), applied to 8 frames of every cat-85 clip, each cut out with the set's own outline.
GPU for speed (minutes).
usage: python clip_identity.py REAL_DIR FRAMES_DIR PIKA_FRAMES_DIR PIKA_ANN_DIR OUT_JSON   (env FRAMES2: a second frames folder)"""
import json
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cat_identity import embed, load_model, other_cats, square_cut  # noqa: E402

real, frames_dir, pika_frames, pika_ann, out = sys.argv[1:6]
dev = "cuda" if torch.cuda.is_available() else "cpu"
model = load_model(dev)
base = os.path.join(real, "database", "processed", "Annotations", "Full-Resolution")
kept = json.load(open(os.path.join(real, "kept_frames.json")))


def cuts_for(folder, ann_dir, names, n=8):
    """n frames spread over the clip, each cut out with its outline (the 256 px full-frame outline, enlarged)"""
    M = np.load(os.path.join(ann_dir, "full-256.npy"))[..., 0]
    out_ = []
    for q in np.linspace(0.05, 0.95, n):
        k = int(round(q * (len(names) - 1)))
        fp = os.path.join(folder, "%05d.jpg" % names[k])
        if not os.path.exists(fp):
            continue
        img = Image.open(fp).convert("RGB")
        m = np.asarray(Image.fromarray(M[k].astype(np.uint8) * 255).resize(img.size, Image.NEAREST)) > 0
        c = square_cut(np.asarray(img), m)
        if c is not None and m.mean() > 0.003:
            out_.append(c)
    return out_


pn = sorted(int(f[:5]) for f in os.listdir(pika_frames) if f.endswith(".jpg") and f[:5].isdigit())
T = embed(model, cuts_for(pika_frames, pika_ann, pn, n=24), dev)
template = T.mean(0)
template /= np.linalg.norm(template)
_, imp = other_cats(model, dev)
thr = float(np.percentile(imp @ template, 99))
own = T @ template
print("template: %d frames of the Body tab clip; their own similarity median %.3f (min %.3f); threshold %.3f" % (
    len(T), np.median(own), own.min(), thr), flush=True)
res = {"threshold": thr, "template_frames": len(T), "template_self_median": float(np.median(own)), "clips": {}}
embs = {"body": T}
for s in sorted(os.listdir(base)):
    if s not in kept:
        continue
    folder = os.path.join(frames_dir, s)
    if not os.path.isdir(folder) and os.environ.get("FRAMES2"):
        folder = os.path.join(os.environ["FRAMES2"], s)
    if not os.path.isdir(folder):
        continue
    cs = cuts_for(folder, os.path.join(base, s), kept[s])
    if not cs:
        continue
    E = embed(model, cs, dev)
    embs[s[-4:]] = E
    sim = E @ template
    res["clips"][s[-4:]] = {"frames": len(sim), "median": float(np.median(sim)), "accepted_share": float((sim >= thr).mean()),
                           "other_cats_more_similar_share": float(np.median([(imp @ template >= v).mean() for v in sim]))}
    print("%s  frames %d  similarity median %.3f  accepted %.0f%%" % (s[-4:], len(sim), np.median(sim), 100 * (sim >= thr).mean()), flush=True)
json.dump(res, open(out, "w"), indent=1)
np.savez(out.replace(".json", "_emb.npz"), **embs)  # every clip's frame embeddings, for clip-to-clip comparison
