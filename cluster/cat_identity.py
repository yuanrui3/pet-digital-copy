"""Identity check for the cat test (work/pets/PROTOCOL_CAT.md, "Identity check"): is the reconstruction this cat,
rather than just a cat? MegaDescriptor-L-384 embeddings; template = mean of the cat's used real frames; threshold
= 99th percentile of the Oxford-IIIT Pet cats' similarities to that template.

usage: python cat_identity.py EXP_DIR [--trial TRIAL_DIR]
EXP_DIR holds frames.npz from cat_eval.py; writes EXP_DIR/identity.json and EXP_DIR/lookalikes.jpg.
"""
import argparse
import json
import os

import numpy as np
import timm
import torch
import torch.nn.functional as F
from PIL import Image

D = os.environ.get("PETS_ROOT") or "/scratch/CLUSTER/%s/claude-pets" % os.environ["USER"]
MODEL = os.path.join(D, "models/megadescriptor-L-384/pytorch_model.bin")
PETS = os.path.join(D, "data/oxford-pets")
CACHE = os.path.join(D, "cache/identity/oxford_cats_megaL384.npz")
SIZE = 384


def square_cut(img, mask, margin=0.1):
    """img (H,W,3) uint8, mask (H,W) bool -> (SIZE,SIZE,3) uint8 cut out on black, square around the outline"""
    if not mask.any():
        return None
    img = img * mask[..., None].astype(img.dtype)
    ys, xs = np.nonzero(mask)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    side = max(int(max(y1 - y0, x1 - x0) * (1 + margin)), 1)
    sy, sx = (y0 + y1) // 2 - side // 2, (x0 + x1) // 2 - side // 2
    canvas = np.zeros((side, side, 3), np.uint8)
    a0, b0 = max(sy, 0), max(sx, 0)
    a1, b1 = min(sy + side, img.shape[0]), min(sx + side, img.shape[1])
    canvas[a0 - sy:a1 - sy, b0 - sx:b1 - sx] = img[a0:a1, b0:b1]
    return np.asarray(Image.fromarray(canvas).resize((SIZE, SIZE), Image.BILINEAR))


def load_model(device):
    model = timm.create_model("swin_large_patch4_window12_384", pretrained=False, num_classes=0)
    state = torch.load(MODEL, map_location="cpu", weights_only=True)
    # the published file is a training checkpoint: {"model": weights, "objective": ..., "optimizer": ...}
    if isinstance(state.get("model"), dict):
        state = state["model"]
    state = {k.removeprefix("module."): v for k, v in state.items()}
    # saved with an older timm whose Swin layout puts each patch-merging layer at the end of a stage; timm's own
    # converter (what timm.create_model("hf-hub:BVRA/...") applies when it loads the file) moves them
    from timm.models.swin_transformer import checkpoint_filter_fn
    state = checkpoint_filter_fn(state, model)
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert not missing, "missing weights: %s" % missing[:5]
    if unexpected:
        print("ignored weights:", unexpected[:5])
    return model.to(device).eval()


@torch.no_grad()
def embed(model, images, device, batch=16):
    """images: list of (SIZE,SIZE,3) uint8 -> (N,1536) unit vectors (the model card's transform: /255, mean/std 0.5)"""
    out = []
    for i in range(0, len(images), batch):
        x = torch.tensor(np.stack(images[i:i + batch]), device=device).permute(0, 3, 1, 2).float() / 255
        out.append(F.normalize(model((x - 0.5) / 0.5), dim=-1).cpu())
    return torch.cat(out).numpy()


def other_cats(model, device, limit=0):
    cache = CACHE if not limit else CACHE.replace(".npz", "_test%d.npz" % limit)
    if os.path.isfile(cache):
        c = np.load(cache)
        return c["names"].tolist(), c["emb"]
    names, cuts = [], []
    with open(os.path.join(PETS, "annotations/list.txt")) as f:
        rows = [r.split() for r in f if not r.startswith("#")]
    skipped = 0
    for name, _cls, species, _breed in rows:
        if species != "1":  # 1 = cat, 2 = dog
            continue
        if limit and len(names) >= limit:
            break
        try:
            img = np.asarray(Image.open(os.path.join(PETS, "images", name + ".jpg")).convert("RGB"))
            tri = np.asarray(Image.open(os.path.join(PETS, "annotations/trimaps", name + ".png")))
            # 1 = animal, 2 = background, 3 = a border band that includes background; the animal alone, as tight as
            # the cat's own masks (the border band left coloured fringes, which make other cats look less cat-like)
            cut = square_cut(img, tri == 1)
        except Exception:
            cut = None
        if cut is None:
            skipped += 1
            continue
        names.append(name)
        cuts.append(cut)
    print("other cats: %d photos, %d skipped" % (len(names), skipped))
    emb = embed(model, cuts, device)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    np.savez(cache, names=np.array(names), emb=emb)
    return names, emb


def cuts_of(frames, which, idx):
    img, mask = frames[which], frames[which + "_mask"]
    return [square_cut(img[i], mask[i]) for i in idx]


def sims(model, device, cuts, template):
    """cosine to the template; a frame with nothing drawn gets -1 (never accepted)"""
    ok = [i for i, c in enumerate(cuts) if c is not None]
    s = np.full(len(cuts), -1.0)
    if ok:
        s[ok] = embed(model, [cuts[i] for i in ok], device) @ template
    return s


def summary(s, thr, imp):
    return {
        "accepted_share": float((s >= thr).mean()),
        "accepted": int((s >= thr).sum()),
        "frames": len(s),
        "median_similarity": float(np.median(s)),
        "share_of_other_cats_more_similar_median": float(np.median([(imp >= v).mean() for v in s])),
        "per_frame": [round(float(v), 4) for v in s],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exp")
    ap.add_argument("--trial", default="")
    ap.add_argument("--test_limit", type=int, default=0, help="testing only: use the first N other cats")
    ap.add_argument("--out_name", default="identity.json")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(device)
    fr = np.load(os.path.join(args.exp, "frames.npz"))
    held, used = fr["held"], fr["used"]

    template = embed(model, [c for c in cuts_of(fr, "real", used) if c is not None], device).mean(0)
    template /= np.linalg.norm(template)
    names, imp_emb = other_cats(model, device, args.test_limit)
    imp = imp_emb @ template
    thr = float(np.percentile(imp, 99))

    out = {
        "model": "MegaDescriptor-L-384",
        "other_cats": len(names),
        "threshold_99th_percentile": thr,
        "other_cats_median_similarity": float(np.median(imp)),
        "real_held_back": summary(sims(model, device, cuts_of(fr, "real", held), template), thr, imp),
        "renders_held_back": summary(sims(model, device, cuts_of(fr, "drawn", held), template), thr, imp),
        "renders_given_views_context": summary(sims(model, device, cuts_of(fr, "drawn", used), template), thr, imp),
    }
    if args.trial:
        tr = np.load(os.path.join(args.trial, "frames.npz"))
        assert (tr["held"] == held).all()
        out["trial_renders_held_back"] = summary(sims(model, device, cuts_of(tr, "drawn", held), template), thr, imp)
        out["informative"] = (out["real_held_back"]["accepted_share"] >= 0.9
                              and out["trial_renders_held_back"]["accepted_share"] <= 0.1)
    else:
        out["informative"] = None  # decided only with the trial's renders as the floor
    with open(os.path.join(args.exp, args.out_name), "w") as f:
        json.dump(out, f, indent=1)

    # the six other cats most like this cat, beside two of its own used frames (a look at what "similar" means)
    top = np.argsort(-imp)[:6]
    own = [c for c in cuts_of(fr, "real", used[:: max(1, len(used) // 2)][:2]) if c is not None]
    tiles = own + [square_cut(np.asarray(Image.open(os.path.join(PETS, "images", names[i] + ".jpg")).convert("RGB")),
                              np.isin(np.asarray(Image.open(os.path.join(PETS, "annotations/trimaps", names[i] + ".png"))),
                                      [1])) for i in top]
    Image.fromarray(np.concatenate([np.asarray(Image.fromarray(t).resize((192, 192))) for t in tiles], 1)).save(
        os.path.join(args.exp, args.out_name.replace("identity", "lookalikes").replace(".json", ".jpg")), quality=90)

    for k in ("real_held_back", "renders_held_back", "renders_given_views_context", "trial_renders_held_back"):
        if k in out:
            print("%-30s accepted %d/%d, median similarity %.3f" % (
                k, out[k]["accepted"], out[k]["frames"], out[k]["median_similarity"]))
    print("threshold %.3f (99th percentile of %d other cats); informative: %s" % (thr, len(names), out["informative"]))


if __name__ == "__main__":
    main()
