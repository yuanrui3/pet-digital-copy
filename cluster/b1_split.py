"""Behaviour test B1 (work/pets/PROTOCOL_BEHAVIOUR.md): recover the videos in Agent-to-Sim's released training file
and hold back the last two.

The file holds every 64-frame window (stride 1) of each video, video after video, without video labels. Two
neighbouring windows belong to the same video exactly when the second is the first shifted by one frame.

usage: python b1_split.py IN.pkl OUT_DIR LOAD_LOGNAME
writes OUT_DIR/LOAD_LOGNAME-train-L64-S1.pkl (training videos, every window),
       OUT_DIR/LOAD_LOGNAME-heldout-L64-S10.pkl (held-back videos, a window every 10 frames = 1 s),
       OUT_DIR/split.json
"""
import json
import os
import pickle
import sys

import numpy as np

KEYS = ("poses", "joints3D", "se3", "cam_se3")
HOLD_BACK = 2  # the last two videos in the file's order, fixed by the protocol
TEST_STRIDE = 10


def same_video(d, i):
    """window i+1 is window i shifted by one frame"""
    return all(np.array_equal(np.asarray(d[k][i])[1:], np.asarray(d[k][i + 1])[:-1]) for k in KEYS)


def main(src, out_dir, name):
    with open(src, "rb") as f:
        d = pickle.load(f)
    n = len(d["se3"])
    assert all(len(d[k]) == n for k in KEYS), {k: len(d[k]) for k in KEYS}
    starts = [0] + [i + 1 for i in range(n - 1) if not same_video(d, i)]
    videos = [(s, e) for s, e in zip(starts, starts[1:] + [n])]  # window index ranges
    lengths = [e - s + 63 for s, e in videos]  # frames per video (windows + 63)
    held = list(range(len(videos) - HOLD_BACK, len(videos)))
    train_idx = [i for v, (s, e) in enumerate(videos) if v not in held for i in range(s, e)]
    test_idx = [i for v in held for i in range(videos[v][0], videos[v][1], TEST_STRIDE)]

    os.makedirs(out_dir, exist_ok=True)
    for idx, path in ((train_idx, "%s-train-L64-S1.pkl" % name), (test_idx, "%s-heldout-L64-S10.pkl" % name)):
        with open(os.path.join(out_dir, path), "wb") as f:
            pickle.dump({k: [d[k][i] for i in idx] for k in KEYS}, f)
    info = {
        "source": src,
        "windows_in_file": n,
        "videos_found": len(videos),
        "frames_per_video": lengths,
        "seconds_per_video": [round(x / 10, 1) for x in lengths],
        "held_back_videos_file_order": held,
        "train_windows": len(train_idx),
        "heldout_windows_every_10_frames": len(test_idx),
        "heldout_share_of_windows": round(sum(videos[v][1] - videos[v][0] for v in held) / n, 4),
    }
    with open(os.path.join(out_dir, "split.json"), "w") as f:
        json.dump(info, f, indent=1)
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:4])
