# Pet copy, first test: the 3D look of one animal from one casual video

Written on 2026-11-14, before any reconstruction was run. The pass rule below must not change once results exist.

## Question

Can released, open code rebuild an individual pet from one casual video well enough that views it was
never given look like that animal? This is the "look" part of the pet copy only. Behaviour, the diary
and "how far the copy leans toward the average animal" are not tested here.

## Method and footage

| What | Source | Terms |
| --- | --- | --- |
| Reconstruction | 4D-Animal (WACV 2026), `github.com/zhongshsh/4D-Animal` at commit `2b8a959` | Apache-2.0 file; parts derive from Meta's Animal Avatar (CC BY-NC 4.0), so treat as non-commercial |
| Footage | CoP3D (Common Pets in 3D): owners' phone videos walking around their pet; the 50-video subset packed by the 4D-Animal authors (`huggingface.co/datasets/zhongshsh/4D-Animal`, one 30.8 GB archive, not gated) | CC BY-NC 4.0 |
| Body model | SMAL dog model from BITE (MPI), shipped inside that archive | non-commercial research |
| Renderer | Lightplane (Meta), `github.com/facebookresearch/lightplane` | as published |

Dogs only: the body model is a dog model. A cat needs a different tool (Lab4D, MIT licence) and is a later test.

Nothing is tuned. Settings are the authors' defaults at that commit: the first 200 frames of the video,
256 px, 10,000 steps.

## Animals

Fixed by rule, not by looking: the first three sequence ids in the code's own list (`config/keys.py`,
`subsetcop3d`): `1030_23106_17099`, `1050_43243_38827`, `1022_15133_10069`. If one is missing from the
archive or cannot be loaded, the next id in that list replaces it, and the swap is recorded here.

## Two ways of holding frames back

Each uses 150 frames to build and 50 to test.

- **A, interleaved (the authors' protocol):** 15 frames in, 5 held back, repeated. A held-back frame is at
  most 3 frames from one that was used. This reproduces the published setting.
- **B, a held-back block:** frames 75 to 124 are held back as one block (`train_split=75 test_split=50`).
  A held-back frame is up to 25 frames from the nearest one used. This is the test that matters: views
  that are not next to a given picture.

What "held back" means here: a held-back frame's colours never enter the fit. Its pixels are still read
once to estimate the animal's pose in that frame (the method predicts pose from image features), so pose
is not held back, only appearance and shape.

Two places where the code looks at held-back frames during training (read in the code, 2026-11-14,
before any run): (1) every 1,000 steps it scores them and saves a separate "best" copy; the scoring
here loads the regular end-of-training model from `checkpoints/`, never `checkpoints_best/`. (2) Its
early-stop rule ends training if the held-back frames' silhouette IoU falls 0.3 below its best so far,
which only happens if a run collapses. It is left as the authors wrote it; each run's log is searched
for "Early stop" and any run where it fired is reported as such.

## Measures

Computed by the authors' own evaluators on the held-back frames: silhouette overlap (IoU), colour error
inside the animal's outline (PSNR in mask), perceptual difference (LPIPS, lower is better).

Baseline, scored with the same formulas: the nearest frame in time that was used for building, shown as
is. If the reconstruction does not beat simply showing that picture, it has added nothing for that view.

Also reported: how far each held-back view is turned from its nearest used view (degrees, from a rigid
alignment of the fitted body in camera coordinates), and a turntable of the result.

## Pass rule

Decided on arm B only.

- A dog passes if, averaged over its 50 held-back frames, the reconstruction beats the baseline on both
  LPIPS (lower) and silhouette IoU (higher).
- The test passes if at least 2 of the 3 dogs pass.

Reported without a rule: arm A (to compare with the paper's numbers), PSNR in mask for both arms, and the
turntable. Anything I say about the turntable is my judgement by eye and is labelled as that.

## What a pass would and would not mean

- It would mean the method carries an individual dog's look across a gap of views in the same video.
- It would not show the copy is recognisably that dog rather than a similar dog, nor how it looks from a
  side the video never showed, nor anything about footage where the animal is small at the edge of the frame.

## GPU budget

Six runs (three dogs, two arms), one A10 each, at most two at a time on the shared allocation. A short
smoke run comes first to measure the time per run; if six full runs would exceed about 12 GPU-hours in
all, arm A is cut to one dog and that change is recorded here before the runs start.

## Changes

**2026-11-14, after two short trial runs on dog 1, arm B (jobs 12377506: 300 steps; 12377835: 30 steps), before any full run.**
Trial numbers are not results: the model had 3% and 0.3% of its training.

- Memory: the first trial ran out of GPU memory in its final scoring (fragmentation on the 22 GB A10).
  Fix: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. It changes memory handling only, not the method.
- Time: 1.96 s per step, so a full run is about 5.6 GPU-hours, not about 2. Six runs would be about 33
  GPU-hours; the cut written above (arm A on one dog) still leaves about 22. How many runs to make is
  left to the user, because the GPUs are a shared group allocation.
- The silhouette half of the pass rule is weaker than intended. The untrained starting model already beat
  the baseline on silhouette IoU (0.641 against 0.587 on the held-back block) because the code estimates
  each held-back frame's pose and camera from that frame. The rule stays as written; LPIPS is the half
  that carries the test (untrained 0.197, baseline 0.156).
- Added, labelled secondary (not part of the pass rule): colour inside the real outline (PSNR in mask)
  against the baseline, and the share of the 50 held-back frames where the reconstruction beats the
  baseline on LPIPS. Both are reported for every run whatever they show.
- Runs (the user's decision, 2026-11-14: "Three dogs, two at once"): arm B on all three dogs, at most
  two at a time, about 17 GPU-hours. Arm A is not run; nothing in the pass rule depends on it.
- Turntable: my first version turned the animal about the cameras' average "up" and tumbled it end over
  end. It now circles a camera around the animal; "up" is the normal of the plane through the video's
  camera centres. Display only; no score depends on it.
