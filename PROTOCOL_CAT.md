# Pet copy, cat test C1: the 3D look of one cat from one casual video

Written on 2026-11-14, before any cat reconstruction was run. The pass rule below must not change once
results exist. Companion to `PROTOCOL.md` (the dog test); the question, the baseline and the measures
are the same, so the two can be read side by side.

## Question

Can released, open code rebuild an individual cat from one casual video well enough that views whose
colours it was never given look like that cat? The look only; behaviour is not tested.

## Method and footage

| What | Source | Terms |
| --- | --- | --- |
| Reconstruction | Lab4D, `github.com/lab4d-org/lab4d` at commit `e629766` (2025-02-02) | MIT |
| Footage | `cat-pikachu-0`: one phone video of the lead author's cat, pre-processed by the authors (frames, masks, cameras, depth, flow, DINOv2 features); `database/` zip linked from Lab4D's `docs/source/data_models.rst` (917,869,159 bytes) | not stated in Lab4D; the same cat's earlier videos in BANMo are CC BY-NC 4.0, so treat as non-commercial research |

Settings are the authors' for their published single-video cat model (`fg-skel-b120`): foreground only,
quadruped skeleton (`--fg_motion skel-quad`), 120 rounds of 200 steps, 256 px crops. Nothing is tuned.

Software differences from the authors' environment, needed to run on the cluster and not expected to
change results: Python 3.10 and PyTorch 2.4.1 / CUDA 12.1 instead of 3.9 and 2.0.0 / 11.7, and the
quaternion CUDA extension compiled as C++17 instead of C++14 (PyTorch 2.4 requires it).

## What is held back

Lab4D drops near-static frames before training; call the remaining frames 0..N-1 in time order. Frames
from floor(0.375 N) to floor(0.625 N) - 1 are the held-back block: the middle quarter, the same place
and share as the dog test's frames 75-124 of 200. Fixed by rule before the data were looked at.

Only colour is held back. Lab4D learns each frame's camera, body pose and lighting code as smooth
functions of time, so a frame left out of training entirely would get an interpolated pose, and the
score would mostly measure pose error. Instead the colour loss is set to zero on held-back frames (a
one-line change after `loss_dict["rgb"]` in `lab4d/engine/model.py`, applied by
`cluster/lab4d_heldout.patch`); their outlines, optical flow, depth and DINOv2 features still train the
model. So, as with the dogs, pose is not held back; unlike the dogs, shape is not held back either,
which makes the silhouette half of the rule even easier than it was there. The DINOv2 features (16
dimensions after PCA) carry coarse appearance information from the held-back frames; colour does not.
The per-frame lighting code at a held-back frame is whatever the time function gives from its
neighbours.

Lab4D never scores held-back frames during training, keeps no "best" copy and has no early stop; the
model scored is the last one (`ckpt_latest.pth`).

## Measures

The same formulas as the dog test (4D-Animal's `iou`, `psnr`, LPIPS with VGG), computed on Lab4D's own
256 x 256 crop around the cat for every held-back frame:

- silhouette overlap (IoU) between the rendered mask (> 0.5) and the real mask;
- colour error inside the real outline (PSNR in mask);
- perceptual difference (LPIPS, lower is better) between the real frame cut out by its mask and the
  rendering on black.

Baseline, scored with the same formulas: the nearest frame in time that was used, cut out by its own
mask and shown as is. The crops are centred on the cat, so this baseline is harder to beat than in the
dog test, where frames were not centred.

Also reported: how far each held-back view is turned from its nearest used view (degrees between the
two frames' fitted object-to-camera rotations; body pose not included), and a turntable rendered with
Lab4D's own `render.py --viewpoint rot-0-360` at a used frame.

## Identity check (added 2026-11-14 at the user's request, before any cat run; no pass rule)

The pass rule above cannot tell this cat from a similar cat. This check asks that question, in the way the
influencer project's face verifier asked "same person or a look-alike?".

- Model: MegaDescriptor-L-384 (`huggingface.co/BVRA/MegaDescriptor-L-384`, CC BY-NC 4.0), an open
  animal re-identification model, used as its model card shows (384 x 384, mean and spread 0.5). It was
  trained on wildlife datasets and probably never saw individual pet cats (the one public cat-identity
  set, CatIndividualImages, sits behind a Kaggle login and was added to its dataset list after the model
  was released), so whether it can tell cats apart at all is checked below, not assumed.
- Other cats: every cat photo in the Oxford-IIIT Pet dataset (`robots.ox.ac.uk/~vgg/data/pets/`,
  CC BY-SA 4.0; about 2,400 photos of 12 breeds), cut out with the dataset's own outlines.
- Every image is cut out on black by its mask (the real outline, or the reconstruction's drawn outline),
  cropped to a square around the outline with a 10% margin, and embedded. Similarity is the cosine
  between an image's embedding and the cat's template: the normalised mean embedding of all used frames.
- Threshold: the 99th percentile of the other cats' similarities (1 in 100 other cats accepted).
- Reported: the share of held-back renders accepted, their median similarity, and each render's rank
  among the other cats; the same for the real held-back frames (the best the measure can show).
- Validity check, fixed now: the measure counts as informative only if the real held-back frames are
  accepted at least 9 times in 10 and the 2-round trial's renders (an almost untrained model) at most 1
  time in 10. Otherwise it is reported as not informative, with the numbers.

## Pass rule

- The cat passes if, averaged over its held-back frames, the reconstruction beats the baseline on both
  LPIPS (lower) and silhouette IoU (higher).
- One cat, so C1 passes if the cat passes.

Reported without a rule, whatever they show: PSNR in mask against the baseline, the share of held-back
frames where the reconstruction wins on LPIPS, the turn angles. Anything said about the turntable is my
judgement by eye and is labelled as that.

## What a pass would and would not mean

- It would mean the method carries one cat's colouring across a gap of views in the same video.
- On its own it would not show the copy is recognisably that cat rather than a similar cat (the
  identity check above speaks to that, within its limits), nor how it looks from
  a side the video never showed, nor anything about several videos taken on different days (that is
  test C2, not run).

## GPU budget

At most two of the group's GPUs at a time, counting the dog runs. A short trial (2 rounds) first, to
measure time and check the pipeline end to end; then one full run (estimate 3-4 A10-hours, from the
authors' 21 minutes per 20 rounds on an RTX 3090). If the trial shows a full run would take more than
8 hours, I stop and ask before running it.

## Changes

**2026-11-14, 21:30, after the first trial (job 12385174) failed and before any result existed.**
Lab4D's learning-rate schedule warms up over 2 rounds (`pct_start = 2 / num_rounds`), so a 2-round run
has no decay phase and fails with a division by zero at its last step. The trial is therefore 4 rounds
(800 of the full run's 24,000 steps, about 3%, the same share as the dogs' 300-step trial), and the
identity check's floor is that 4-round trial's renders wherever this protocol says "2-round trial".
What the failed trial did show: colour was held back on 19 of 76 frames (raw ids 164..281), the first
batch zeroed 69 of 256 samples, and one round took 169 s, so a full run is about 5-6 hours, under the
8-hour limit above.

**2026-11-14, 22:40, before any identity result on the full run.** The identity script's first test
(CPU, the 4-round trial's frames, only 20 other cats; test numbers, not results: real held-back frames
19/19 accepted, trial renders 0/19) showed that cutting the other cats out with the Oxford outline's
"border" band leaves a coloured fringe of background around them, which makes them look less cat-like and
would lower the threshold in the renders' favour. Other cats are now cut out with the "animal" class
only, as tight as the cat's own masks. Two loading fixes, no change to the measure: the published model
file is a training checkpoint (weights under "model"), saved with an older timm layout that timm's own
converter (`checkpoint_filter_fn`) maps.

**2026-11-15, 07:40, after C1's result (job 12385380: passes), at the user's request ("Yes, add it"),
before this comparison was run.** The nearest used frame in time is turned up to 109 degrees from the
held-back view (median 43), so it is a weak stand-in in the middle of the block. Added one more comparison,
reported without a rule: for each held-back frame, the used frame nearest in viewpoint (smallest angle
between the two frames' fitted object-to-camera rotations, the same angle as "turn" above; body pose not
included), chosen among all used frames, cut out and scored exactly as the baseline is (same crop, mask
and measures; `cluster/cat_view_baseline.py`, which reuses `cat_eval.py`'s functions and re-checks that it
reproduces the baseline's numbers). The pass rule and C1's verdict stay as they are, against the nearest
frame in time.

**2026-11-15, 09:30, after the viewpoint comparison (job 12388306), at the user's request ("Yes, add it"),
before this measure was run.** The viewpoint comparison showed that every measure so far (LPIPS, IoU, PSNR
in mask) rewards the copy for being drawn in the held-back frame's exact pose and place, which a photo
from another moment cannot match; LPIPS barely moves between given and held-back frames (0.200 vs 0.216)
although the copy's flank visibly darkens. Added a coat-colour measure that ignores pose and position,
reported without a rule (`cluster/cat_colour.py`, CPU only, from `frames.npz`):

- Pixels: each cat's own cut-out (real frame: its mask; copy: its drawn mask; photo: its mask), with 3
  pixels taken off every mask's edge so pixels blended with the black background do not count; converted
  from sRGB to CIELAB (D65).
- Measures, both in CIELAB units (about 2 = barely visible difference): the distance between the real
  cat's and the shown cat's average colour, and a distance between their whole colour spreads (sliced
  Wasserstein: 64 fixed random directions, 100 quantiles each), which also counts blotches that the
  average hides.
- Compared, on the 19 held-back frames: the copy, the nearest photo in time, the nearest photo in
  viewpoint. Context: the copy on the frames whose colour it was given. Also reported: average lightness
  and colourfulness (CIELAB L and chroma) of each.

**2026-11-15, 10:00, after the coat-colour measure (job 12388567: the copy's average colour is 13.0 off on
held-back frames, 0.4 on given ones, worst far in time from any given frame), at the user's request ("Yes,
test it"), before this test was run.** A diagnostic, not a new pass rule, and no retraining. Lab4D's cat
colour depends on a 32-number appearance code that is a small network over time (`appr_embedding`, a
TimeMLP of the raw frame id, in `lab4d/nnutils/nerf.py`); with colour held back, nothing pins that code on
held-back frames. Test (`cluster/cat_appr_swap.py`): re-draw the 19 held-back frames from the same trained
model, same pose and camera, changing only the appearance code they get:

- unchanged (must reproduce the copy's earlier numbers; a check that nothing else moved);
- the code of the nearest given frame in time;
- the code of the nearest given frame in viewpoint;
- Lab4D's own average code (`get_mean_vals`).

Scored with the coat-colour measure above and with LPIPS, IoU and PSNR in mask. Read-out fixed now, on the
average-colour difference with the nearest-in-time code against the unchanged copy: if it falls by at
least half, the drifting appearance code is the main cause of the darkening; by less than a quarter, it is
not; in between, it is part of the cause.

## Second cat (demo only, no test)

**2026-11-16, 00:40.** Asked for by the user ("Build a second cat, also explore stimuli response"; they chose
Lab4D's 85-cat set). Not a scored test: nothing was held back, and nothing here has a pass rule.

- Video: `cat-85-0068` from Lab4D's `cat-85` set (RAC, Yang et al. 2023), a grey cat with white paws and
  chest, standing on a chair and jumping down to the floor; 225 raw frames, 133 with masks used for
  training. Folders `cat-85-0055`..`0084` are Pikachu itself and were skipped.
- Training: Lab4D `fg-skel` with `skel-quad`, 60 rounds (half of Pikachu's), job 12393539 (21:43 to 00:24, about 2.7 GPU-hours),
  `cluster/run_cat2.sbatch cat85-0068 60 2`. Output `experiments_cat2/cat85-0068-fg-skel-r60`.
- Viewer export: 113 frames (every 2nd raw frame), 22,968 vertices, 43,999 faces; UV bake at 2048 px;
  photo coat from the 133 training frames (`cat_tex_photo_detail_loose`).
- Photo-coat checks (from `photo.json`): model outline vs mask IoU median 0.882, min 0.796 (Pikachu: 0.945);
  98% of texels seen by some frame, 90% within 60° of head-on; 3% keep the network's colour.
- Look, by eye against three raw frames: shape, white paws and chest right; the coat comes out warmer and
  browner than the real cool grey, and softer than Pikachu's because the outline fits less tightly.
- On the page: Pikachu's Keep version 11, Body tab, "Cat" menu → "Grey and white cat".
