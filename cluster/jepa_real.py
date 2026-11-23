"""The video-model judge (work/pets/PROTOCOL_CAT.md, "A video-model judge"): the real cats' silhouette clips.

Per cat-85 clip, raw frames at 30 a second:
  - which clips: all with keypoints in POSE_DIR, rolling clips 0078 to 0080 out;
  - outline: SAM 2.1 small (video), prompted with the set's own Lab4D outline at a kept frame every PROMPT_EVERY raw
    frames (or the nearest kept frame), tracked forwards and backwards; at the other kept frames its outline is
    compared with Lab4D's (IoU, reported);
  - the camera's pan removed: the background's motion between raw frames (Farneback optical flow at quarter size,
    the median outside the cat's widened outline), summed, so the clip is as if from a still camera;
  - clips: 16 frames, one every 4 raw frames (7.5 a second, 2.0 s), starting every 32 raw frames; one still square
    per clip round everything the cat covers (1.15 times the larger side), 512 px reduced to 256 (area averaging), as
    jepa_ours.py draws ours;
  - kept if walking and seen whole, rules fixed before any clip was scored: the body side-on (gait_stats.py's rule)
    on at least half of its raw frames and at least 2 swings (gait_stats.py's: a paw moving forward against the body
    faster than 0.25 body lengths a second for 3 frames or more) starting inside it; an outline on all 16 frames;
    the outline touching the picture's edge on at most 2 of them.
GPU (SAM 2). env-judge.
usage: python jepa_real.py SAM2_CKPT REAL_DIR RAW_DIR POSE_DIR OUT_NPZ [clip ...]
  REAL_DIR: experiments_judge/real (Annotations, kept_frames.json); RAW_DIR: .../JPEGImagesRaw/Full-Resolution"""
import json
import os
import sys
import time

import cv2
import numpy as np
import torch

ckpt, real, raw_dir, pose_dir, out = sys.argv[1:6]
# every clip with keypoints (the 17 of "Real cats' gait from video" and, since those gave too few walking windows,
# the archive's other clips), the rolling clips 0078 to 0080 left out
CLIPS = sys.argv[6:] or sorted(f[:-4] for f in os.listdir(pose_dir) if f.startswith("cat-85-") and f.endswith(".npz")
                               and f[7:11] not in ("0078", "0079", "0080"))
PROMPT_EVERY = int(os.environ.get("PROMPT_EVERY", 15))
NF, EVERY, STEP, RES, MARGIN = 16, 4, 32, 256, 1.15
SPAN = EVERY * (NF - 1)
FPS, SWING_V, SMOOTH = 30.0, 0.25, 5
POSE_NAMES = ["nose", "upper_jaw", "lower_jaw", "mouth_end_right", "mouth_end_left", "right_eye", "right_earbase",
              "right_earend", "right_antler_base", "right_antler_end", "left_eye", "left_earbase", "left_earend",
              "left_antler_base", "left_antler_end", "neck_base", "neck_end", "throat_base", "throat_end", "back_base",
              "back_end", "back_middle", "tail_base", "tail_end", "front_left_thai", "front_left_knee", "front_left_paw",
              "front_right_thai", "front_right_knee", "front_right_paw", "back_left_paw", "back_left_thai",
              "back_right_thai", "back_left_knee", "back_right_knee", "back_right_paw", "belly_bottom",
              "body_middle_right", "body_middle_left"]
I = {n: k for k, n in enumerate(POSE_NAMES)}
LEGS = [("front_left_thai", "front_left_paw"), ("front_right_thai", "front_right_paw"),
        ("back_left_thai", "back_left_paw"), ("back_right_thai", "back_right_paw")]


def smooth(x, k=SMOOTH):
    pad = np.r_[np.repeat(x[:1], k // 2), x, np.repeat(x[-1:], k // 2)]
    return np.convolve(pad, np.ones(k) / k, mode="valid")


def runs(mask):
    d = np.diff(np.r_[0, mask.astype(int), 0])
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def walking(path, nraw):
    """per raw frame: side-on (bool), and the raw frames where a swing starts (gait_stats.py's rules)"""
    z = np.load(path)
    kp, rawi = z["kp"].astype(np.float64), z["raw"]
    side_on, onsets = np.zeros(nraw, bool), []
    a = kp[:, I["neck_base"], :2] - kp[:, I["tail_base"], :2]
    L = np.linalg.norm(a, axis=1)
    ax = a / np.maximum(L, 1e-9)[:, None]
    level = np.degrees(np.arcsin(np.abs(ax[:, 1]))) <= 35
    side = level & (L >= 0.7 * np.percentile(L, 90)) & (kp[:, I["neck_base"], 2] >= 0.5) & (kp[:, I["tail_base"], 2] >= 0.5)
    side_on[rawi[side]] = True
    Ls = smooth(L)
    for gn, pn in LEGS:
        gp, pp = kp[:, I[gn]], kp[:, I[pn]]
        ok = side & (gp[:, 2] >= 0.5) & (pp[:, 2] >= 0.5)
        u = smooth(((pp[:, :2] - gp[:, :2]) * ax).sum(1) / Ls)
        du = np.gradient(u) * FPS
        for s, e in runs(du > SWING_V):
            if e - s >= 3 and ok[s:e].all():
                onsets.append(int(rawi[s]))
    return side_on, np.array(sorted(onsets))


def kept_mask(M, K, k, hw):
    """the Lab4D outline of kept frame k in the half-size raw picture"""
    m = (M[k, ..., 0] & M[k, ..., 1]).astype(np.uint8) * 255
    A = np.array([[K[k, 0], 0, K[k, 2]], [0, K[k, 1], K[k, 3]]], np.float64) * 0.5
    return cv2.warpAffine(m, A, (hw[1], hw[0]), flags=cv2.INTER_LINEAR) > 127


from sam2.build_sam import build_sam2_video_predictor  # noqa: E402

dev = "cuda"
pred = build_sam2_video_predictor("configs/sam2.1/sam2.1_hiera_s.yaml", ckpt, device=dev)
pred.max_cond_frames_in_attn = 6
kept = json.load(open(os.path.join(real, "kept_frames.json")))
ann = os.path.join(real, "database", "processed", "Annotations", "Full-Resolution")
clips, meta, t_all = [], [], time.time()
for clip in CLIPS:
    t0 = time.time()
    fdir = os.path.join(raw_dir, clip)
    names = sorted(os.listdir(fdir), key=lambda s: int(os.path.splitext(s)[0]))
    nraw = len(names)
    first = cv2.imread(os.path.join(fdir, names[0]))
    hw = (first.shape[0] // 2, first.shape[1] // 2)
    M = np.load(os.path.join(ann, clip, "crop-256.npy"))
    K = np.load(os.path.join(ann, clip, "crop-256-crop2raw.npy"))
    det = np.load(os.path.join(ann, clip, "crop-256-is_detected.npy"))
    rid = kept[clip]
    good = [k for k in range(len(rid)) if rid[k] < nraw and det[k] and (M[k, ..., 0] & M[k, ..., 1]).sum() >= 400]
    # prompts: the good kept frame nearest each multiple of PROMPT_EVERY
    prompts = sorted({min(good, key=lambda k: abs(rid[k] - t)) for t in range(0, nraw, PROMPT_EVERY)})
    masks = np.zeros((nraw,) + hw, bool)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        st = pred.init_state(video_path=fdir, offload_video_to_cpu=True)
        for k in prompts:
            pred.add_new_mask(st, frame_idx=int(rid[k]), obj_id=1, mask=kept_mask(M, K, k, hw))
        for rev in (False, True):
            for fi, _, logits in pred.propagate_in_video(st, reverse=rev):
                m = (logits[0, 0] > 0).float().cpu().numpy()
                masks[fi] = cv2.resize(m, (hw[1], hw[0]), interpolation=cv2.INTER_LINEAR) > 0.5
        pred.reset_state(st)
    del st
    torch.cuda.empty_cache()
    used = {int(rid[k]) for k in prompts}
    ious = []
    for k in good:
        if int(rid[k]) in used or rid[k] >= nraw:
            continue
        a, b = masks[rid[k]], kept_mask(M, K, k, hw)
        ious.append((a & b).sum() / max((a | b).sum(), 1))
    # the camera's pan: background motion between raw frames, summed (half-size pixels)
    small = (hw[1] // 2, hw[0] // 2)
    prev, cum = None, np.zeros((nraw, 2))
    for f in range(nraw):
        g = cv2.cvtColor(cv2.resize(cv2.imread(os.path.join(fdir, names[f])), small, interpolation=cv2.INTER_AREA),
                         cv2.COLOR_BGR2GRAY)
        if prev is not None:
            fl = cv2.calcOpticalFlowFarneback(prev, g, None, 0.5, 4, 21, 3, 5, 1.1, 0)
            cat = cv2.dilate(cv2.resize(masks[f - 1].astype(np.uint8), small, interpolation=cv2.INTER_NEAREST),
                             np.ones((15, 15), np.uint8)) > 0
            bg = fl[~cat]
            cum[f] = cum[f - 1] + (2 * np.median(bg, axis=0) if len(bg) > 500 else 0)
        prev = g
    # clips
    side_on, ons = walking(os.path.join(pose_dir, clip + ".npz"), nraw)
    area = masks.reshape(nraw, -1).sum(1)
    edge = masks[:, 0].any(1) | masks[:, -1].any(1) | masks[:, :, 0].any(1) | masks[:, :, -1].any(1)
    nwin = nkept = 0
    for s in range(0, nraw - SPAN, STEP):
        fr = s + EVERY * np.arange(NF)
        nwin += 1
        walk = side_on[s:s + SPAN + 1].mean() >= 0.5 and ((ons >= s) & (ons <= s + SPAN)).sum() >= 2
        seen = (area[fr] >= 0.001 * hw[0] * hw[1]).all() and edge[fr].sum() <= 2
        row = {"clip": clip, "start": int(s), "walking": bool(walk), "seen": bool(seen)}
        if walk and seen:
            lo, hi = np.full(2, np.inf), np.full(2, -np.inf)
            for f in fr:
                r, c = np.nonzero(masks[f])
                xy = np.c_[c, r] - cum[f]  # still-camera coordinates
                lo, hi = np.minimum(lo, xy.min(0)), np.maximum(hi, xy.max(0))
            side = MARGIN * (hi - lo).max()
            cen = (lo + hi) / 2
            clip16 = np.zeros((NF, RES, RES), np.uint8)
            sc = 2 * RES / side
            for j, f in enumerate(fr):
                o = cen - side / 2 + cum[f]  # the square's corner in this frame's pixels
                A = np.array([[sc, 0, -sc * o[0]], [0, sc, -sc * o[1]]])
                big = cv2.warpAffine(masks[f].astype(np.uint8) * 255, A, (2 * RES, 2 * RES), flags=cv2.INTER_LINEAR)
                clip16[j] = cv2.resize(big, (RES, RES), interpolation=cv2.INTER_AREA)
            clips.append(clip16)
            nkept += 1
            row["n"] = len(clips) - 1
        meta.append(row)
    print("%s: %d raw frames, %d prompts, outline IoU against Lab4D median %.3f (%d frames), pan %.0f px; "
          "windows %d, kept %d (%.0f s)" % (clip, nraw, len(prompts), np.median(ious) if ious else np.nan, len(ious),
                                           np.abs(cum[-1]).max(), nwin, nkept, time.time() - t0), flush=True)
    meta.append({"clip": clip, "summary": True, "raw_frames": nraw, "prompts": len(prompts),
                 "iou_median": float(np.median(ious)) if ious else None, "iou_q10": float(np.percentile(ious, 10)) if ious else None,
                 "iou_frames": len(ious), "windows": nwin, "kept": nkept})
rows = [m for m in meta if not m.get("summary")]
np.savez_compressed(out, clips=np.array(clips), clip=np.array([m["clip"] for m in rows if "n" in m]),
                    start=np.array([m["start"] for m in rows if "n" in m]))
json.dump(meta, open(out.replace(".npz", ".json"), "w"), indent=1)
print("real clips kept: %d of %d windows (walking %d, seen whole %d); %.0f s" % (
    len(clips), len(rows), sum(m["walking"] for m in rows), sum(m["seen"] for m in rows), time.time() - t_all))
