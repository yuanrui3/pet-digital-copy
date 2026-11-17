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

## Second cat at home (plan, written 2026-11-16 at 12:20, before running; demo only)

The user picked "Second cat at home". The grey and white cat has no motion of its own at home, so it borrows
Pikachu's: the recorded month and the copy's walks. Its joint angles cannot be copied across: Agent-to-Sim's
cat uses a URDF skeleton with its own joint frames, root orientation and bone centres, while the grey cat was
trained with Lab4D's built-in quadruped skeleton. Instead, skinning transfer:

- fit the grey cat's rest mesh onto Pikachu's (Agent-to-Sim's) rest mesh with one scale, rotation and shift
  (point-to-point ICP both ways, started from the 4 proper principal-axis alignments; keep the closest);
- give each grey-cat vertex the bone weights of its 8 nearest Pikachu vertices (weighted by distance),
  smoothed over the grey cat's mesh, top 4 kept;
- move it with Pikachu's own bone transforms (the page already has them), so it is placed and posed exactly
  like Pikachu, with the grey cat's photo coat.

Check, read-out fixed now: on 40 recorded frames, how much the mesh's edges stretch (|log| of posed over rest
length). "Moves cleanly" if the grey cat's 99th percentile is at most 1.5 times Pikachu's own; "some
stretching" up to 3 times; "tears" beyond. Also reported: fit distance after alignment, and how far the grey
cat's lowest point sits from Pikachu's (floating or sinking), median over the frames.

### Result (2026-11-16, 12:45; CPU only, on the cluster's login node)

- **Skinning transfer, rigid fit** (`cluster/home2_transfer.py`): fit distance 37.2 mm (4.7% of the cat's
  size). By the fixed read-out it "moves cleanly": 99th-percentile stretch 1.23 against Pikachu's own 1.15
  (ratio 1.07); lowest point a median 24 mm above Pikachu's. **But the read-out missed what the page showed**:
  rendered at home (headless Chrome), the grey cat walked plausibly at some moments (video 18 at 90 s) and
  was badly deformed at others (walk 3 at 15 s: stretched thin, head drawn out, legs gone), and its tail did
  not follow Pikachu's. The two rest poses differ (tail, legs, head), so parts pick up the wrong bones. The
  stretch measure did not catch it because Pikachu's own skinning already stretches a lot (15.5% of its edges
  by over 30%), which makes the baseline lenient.
- **Non-rigid fit, then skinning transfer** (`NONRIGID=1`, a smooth displacement field): with weak smoothness
  (down to 1) the fit reached 2.4 mm but reshaped the grey cat (its edges changed length by a median 17%) and
  the mesh tore at home; with strong smoothness only (300, 100) it still tore. Not used.
- **Shipped: coat transfer** (`cluster/home2_coat.py`): at home the grey and white cat is Pikachu's body
  wearing the grey cat's photo coat. Every texel of Pikachu's 2048 px texture takes the colour of the 4
  nearest points of the rigidly fitted grey cat (median 37 mm away, 90th percentile 71 mm). Light texels:
  3.2% against 2.7% on the grey cat itself. It moves exactly as Pikachu does; its shape is Pikachu's, and the
  white paws and chest land roughly, not exactly, in place.
- On the page: version 14 of Pikachu's Keep, "Cat" in the At home bar (swaps the coat on Pikachu's mesh; the
  pale real cat beside a replay stays Pikachu), the note "The grey and white cat at home", and the grey cat's
  Body note updated. Checked by headless-Chrome renders at four moments (recorded video 18 and walk 3).

A proper fix would retarget the motion to the grey cat's own skeleton: map Agent-to-Sim's URDF joints to
Lab4D's built-in quadruped joints and carry over global joint orientations with the grey cat's bone lengths.
Not done.

### Retarget (plan, written 2026-11-16 at 14:30, before running; demo only)

The user picked "Try the retarget". On reading both codes, the two skeletons turn out to be the same 25
joints in the same tree and order: Lab4D's built-in quadruped and Agent-to-Sim's quad URDF (Bauch = spine 1,
Bauch.001 = spine 2, Brust = spine 3, Hals = head, Schalterplatte/Oberarm/Unterarm/Vorderpfote = shoulder to
hand, Schwanz ... end = tail 1 to 5, Oberschenkel/Unterschenkel/Pfote1/Pfote2 = hip to foot). What differs is
the joint frames, rest pose and bone lengths, so joint angles still cannot be copied, but bone i maps to bone
i. Method:

- the grey cat's own rest skeleton, skinning (top 4 bones per vertex) and its own video's bone motion come
  from its Lab4D model (`cluster/home2_skel.py`, one short GPU job), checked against Lab4D's own warp;
- Pikachu's rest joints come from its bone motion on the page: each joint is the point that moves with both
  its bone and its parent bone (least squares over the recorded frames);
- the grey cat's rest mesh and skeleton are lined up with Pikachu's by the same rigid fit as before;
- per bone, a fixed rest-pose correction turns the grey bone's rest direction onto Pikachu's (Kabsch over the
  child directions where a bone has several children, the smallest rotation where it has one; tips copy
  their parent's). Each frame, a grey bone's rotation is Pikachu's matching bone's rotation (rest -> frame)
  times that correction; its joints are placed by forward kinematics with the grey cat's own bone lengths;
  the root joints stay put, as they do in both models. The page does this per frame in the browser from
  Pikachu's bone motion, so only the grey cat's mesh, skinning and the constants are added.
  (`cluster/home2_retarget.py`; the page's `poseGrey`.)

Read-out, fixed now. Stretch: the grey cat's 99th-percentile edge stretch on 40 recorded and 40 walk frames
against its own on 40 frames of its own video (a fairer bar than Pikachu's, which last time proved lenient):
"moves cleanly" up to 1.5 times, "some stretching" up to 3 times, "tears" beyond. Also reported: how far its
lowest point sits from Pikachu's. Ship rule, fixed now because the stretch measure missed the last failure:
headless-Chrome renders at home at recorded video 18 at 90 s and walk 3 at 15 s (the moments that showed the
last failure) and two more, walk 3 at 5 s and recorded video 5 at 20 s. The retarget replaces the coat on the
page only if at all four the grey cat is whole (head, four legs and tail attached, nothing drawn thin or torn)
and stands on or near the floor. Otherwise the coat version stays.

### Retarget result (2026-11-16, 15:10; one 3-minute GPU job, 12400359, the rest CPU on the login node)

- **The grey cat's skeleton export** (`cluster/home2_skel.py`): its rest skeleton rebuilt from the model matches
  Lab4D's own rest bones exactly (to 3e-7). Its 4-bone skinning follows Lab4D's warp on its own video to a
  median of about 0.3% of the body's length (worst 3.6%). Pikachu's joints recovered from its bone motion: residual
  0.06 mm median.
- **The planned method failed.** Bone i following Pikachu's bone i, turned onto Pikachu's bone direction: "some
  stretching" (1.68 times its own video's 99th-percentile stretch), and the body drifted sideways off Pikachu's.
  With its root joints carried by Pikachu's matching bones instead of fixed: still "some stretching" (1.76).
  The same turn from rest, without the direction correction: "moves cleanly" (1.18) by the read-out, but at all
  three moments looked at, the grey cat crouched nose-down with its legs swept back while Pikachu stood. The
  stretch read-out missed this again.
- **Why:** both skeletons were learnt freely, so their bones do not move the same body parts. Grey "L hand" moves
  its right front leg; grey "R shoulder" moves its head and neck; grey "spine 2" moves nothing; Pikachu's
  "spine 2" carries most of its torso and its "R elbow" its upper chest and neck. Also, the grey cat's canonical
  rest pose is bent sideways with its paws tucked together.
- **Used instead (changed after seeing the above):** match by body region, not by bone name
  (`cluster/home2_retarget.py`, MODE=region). The grey cat is first posed as in the frame of its own video
  that best fits Pikachu's rest pose and stands on four clear paws (own frame 50, raw frame 100; fit 23.5 mm,
  against 37.2 mm for its canonical pose). That pose is lined up with Pikachu's rest mesh and bent smoothly onto
  it, with paw tips, nose and tail tip pinned. The bend is used only to find which part of Pikachu each grey
  vertex corresponds to (3.4 mm apart after it) and is never shown. Each grey bone then moves rigidly with the
  blend of Pikachu's bones that move its vertices' corresponding points. Examples: grey L foot follows Pikachu's
  L foot 88%; grey "R shoulder" (head and neck) follows Pikachu's "R elbow" 89%. The page computes this per frame
  from Pikachu's bone motion (`poseGrey`). Each frame it also raises or lowers the grey cat so its lowest point
  sits where Pikachu's does.
- **Read-out:** "moves cleanly", 0.81 times its own video's 99th-percentile stretch. 6.2% of edges stretch by
  over 30%, against 9.8% in its own video. Before the height correction, its lowest point sat a median 32 mm
  below Pikachu's: it is shorter in the leg.
- **Ship rule** (headless renders):
  - Recorded video 18 at 90 s: whole, standing, tail up as Pikachu's.
  - Walk 3 at 5 s: whole.
  - Walk 3 at 15 s, the moment of the earlier failures: head, raised tail and one front leg visible from the
    page's camera. The body looks long from above. The side view of the same frame (home2_retarget.png) shows it
    standing on four legs, so it was judged whole. That judgement leans on the plot, not the page.
  - Recorded video 5 at 20 s: neither cat can be seen, Pikachu included, because the room scan's sofa hides the
    spot (also at 25, 30 and 35 s). It was replaced, after the fact, by recorded video 18 at 40 s (sitting): whole.
  - Passed with that substitution.
- **On the page:** version 15 of Pikachu's Keep. "Cat" in the At home bar now shows the grey and white cat in its
  own body. The coat version (Pikachu's body in the grey coat, version 14) is removed. The note "The grey and
  white cat at home" and the grey cat's Body note are rewritten.
- **Not done:** no measure of how closely its poses follow Pikachu's beyond the pictures; each grey bone moves as a rigid
  blend of several of Pikachu's bones, so sharp joint bends may be softened (not checked).

## Feet and pose check (plan, written 2026-11-16 at 22:20, before any measurement)

The user picked "Feet and pose check" from three directions. The page lines up only the grey cat's single lowest point
with Pikachu's each frame, so its paws may slide or float, and nothing measures how closely its poses follow Pikachu's.
Everything below is measured on what the page shows (version 15 data, posed exactly as the page does it), on all
11,574 recorded frames and all 52,576 walk frames, CPU only (`cluster/feet_check.py`).

- **Landmarks** (each cat's rest mesh, as in the retarget): the four paw tips (the 20 lowest vertices of each paw),
  the nose (20 front-most) and the tail tip (20 highest of the rear fifth); each followed as the mean of its vertices.
- **Contact:** a paw is down when its tip is at most 2 cm above that cat's lowest point (along the room's down) in
  that frame.
- **M1 paw slide** (each cat; recorded and walks apart): a step is a paw down in two frames running (0.1 s); its skid is
  how far the tip moves sideways (across the room's down). Read-out: a cat's paws **slide** if more than 10% of steps
  skid over 1.5 cm in 0.1 s; the grey cat **slides more than Pikachu** if its share is over 1.25 times Pikachu's.
- **M2 floating paws:** for each paw that is down on Pikachu, how high the grey cat's same paw is above the grey cat's
  lowest point. Read-out: **lands with Pikachu** if at most 15% of these are over 3 cm up; otherwise **floating paws**.
- **M3 floor gap** (both cats share it, since the grey cat's lowest point is put at Pikachu's): on frames where
  Pikachu's lowest point is within 10 cm of the room's floor (the page's floor height), its height above the floor.
  Read-out: **on the floor** if the median is within 1.5 cm and the 10th to 90th percentiles within 3 cm; otherwise
  **floats or sinks**.
- **M4 pose closeness:** each landmark's move from its rest place, in the cat's own frame, relative to the cat's centre
  (mean of every third vertex), divided by the cat's rest length from nose to the middle of its hind paws and given in
  Pikachu's centimetres. Error = mean over the six landmarks of the grey cat's move minus Pikachu's, same frame.
  Baseline = the same with Pikachu 5 s later on the same track (a different moment). Read-out (medians): **follows
  closely** if the error is at most a third of the baseline, **loosely** if at most two thirds, otherwise **does not
  follow**.
- **M5 range:** for each landmark and axis, the spread (standard deviation) of the grey cat's moves over that of
  Pikachu's; median over the 18. Read-out: **softened** below 0.8, **exaggerated** above 1.25, otherwise **same range**.
  Also reported per landmark (paws apart from nose and tail tip).
- **Pictures:** side views of the five frames with the largest M4 error, both cats, looked at by eye (the stretch
  measure missed two failures before).

**Fix, only if M1, M2 or M3 says so.** Precomputed per frame and paw, stored with the page data: while a paw is down
(Pikachu's contacts for both cats, so they step together), hold it at the median place of that stretch of contact
(sideways) and at the cat's lowest point (height), easing in and out over 2 frames, capped at 4 cm; the correction
moves the paw and lower leg (weight 1 at the paw tip falling to 0 at 60% of the leg's height in the rest pose). M3
failing: also put the lowest point on the floor when it is within 10 cm of it. **Kept only if** the slide share falls
below 5% and floating paws below 15% on the same measures, the grey cat's stretch stays at most 1.5 times its own
video's (99th percentile), and the ship-rule renders (recorded video 18 at 90 s and 40 s, walk 3 at 5 s and 15 s) look
whole and the paws planted. M4 and M5 are for the record: no fix is planned from them.

### Feet and pose check: result of the measures (2026-11-16, 22:50; `experiments_feet/check`, CPU, 70 s)

Read-outs applied as written:
- **M1 slide: slides**, both cats, recorded and walks. Share of paw-down steps skidding over 1.5 cm in 0.1 s:
  Pikachu 42% recorded / 34% walks; grey cat 38% / 31% (**no more than Pikachu**). So most of the sliding is in
  Pikachu's own motion (the recordings' estimated motion and the copy's walks), not from the retarget.
- **M2: floating paws.** Where Pikachu's paw is down, the grey cat's same paw is over 3 cm up 41% (recorded) / 47%
  (walks) of the time; by paw: +x front 100%, -x hind 51% / 75%, +x hind 17% / 16%, -x front 2% / 0%.
- **M3: floats or sinks.** Frames near the floor: Pikachu's lowest point median 4.2 cm above the page's floor
  (recorded; walks 2.3 cm), 10th to 90th percentiles -4.1 to +8.4 cm (walks -3.6 to +7.7).
- **M4: follows loosely.** Median pose error 3.4 cm against 6.5 cm for Pikachu 5 s later (ratio 0.52; recorded 0.48,
  walks 0.54). Worst five frames: four in recorded video 14 (Pikachu curled up, the grey cat more upright) and one in
  video 18 at 149 s (the grey cat stretched long).
- **M5: same range** (0.94; +x hind 1.34, +x front 0.84, -x hind 0.82).
- **Side check** (`cluster/feet_sides.py`): each grey hind paw moves with Pikachu's same hind paw (bone-mix cosines
  0.95 and 1.00), but both grey front paws move mostly with Pikachu's -x front leg (0.87 and 0.79; with Pikachu's +x
  front leg 0.51 and 0.59).
- Cause of M2 and of the mixed front legs, by the rest landmarks: the grey cat's rest pose (its own video's frame 50)
  has its +x front paw 8 cm up and the hind paws one behind the other (1 cm apart sideways, 10 cm front to back).

**Change, written before running it (22:55):** before the paw fix, re-pick the grey cat's rest frame
(`home2_retarget.py REBASE=level`): of all its own frames, the best fit to Pikachu's rest among those standing square,
the four paws level within 2.5 cm and each pair at least 4 cm apart sideways and at most 5 cm apart front to back
(Pikachu's rest: 1.1 cm, 7.7 and 8.3 cm, 0.4 and 3.4 cm). It is kept if floating paws (M2) fall, the pose error (M4)
grows by no more than a tenth, the stretch read-out still says "moves cleanly", and its renders look whole. Then the
planned paw fix (contacts from Pikachu, paws held while down, lowest point onto the floor near it) on top, judged by the
rule above. Since M3 depends on the floor height, the floor is checked against the room scan under the cat before the
lowest point is moved onto it.

**REBASE=level found no frame (23:00):** I stopped the search after the 79 best-fitting of the grey cat's 113 own
frames (the other 34 fit Pikachu's rest worse); none stood square. Where four paws were found, the most level was
within 3.8 cm but with its pairs 0.2 cm apart sideways; the best with pairs apart was frame 105 (level within 5.5 cm,
6.1 cm apart, 8.7 cm front to back); most were 8 to 12 cm. So the rest pose cannot be fixed by picking another frame.

**Change, written before running it (23:02):** square the current rest pose instead (`cluster/home2_square.py`).
Bend the grey cat's rest mesh as rigidly as possible (ARAP: each vertex's neighbourhood keeps its shape as far as it
can; cotangent weights) so each paw tip moves to a square stance: sideways and front to back, Pikachu's rest paw places
about its centre scaled by the two cats' length ratio (grey 0.58 m, Pikachu 0.53 m); height, the grey cat's lowest paw
now. Held: each paw (its tip and the vertices within 2 cm of it) moved with its tip; the body further than 18 cm from
every paw tip (along the surface) kept in place; the legs in between bend. Then the correspondence and bone mixes are
redone from the squared rest as before (bend with landmarks pinned, region mix). Kept by the rule above (floating paws
fall, M4 grows by no more than a tenth, stretch "moves cleanly", renders whole), plus: the squaring itself stretches
no edge by over 30% on more than 2% of leg edges.

**First squaring try failed (23:10):** Pikachu's stance asked one grey hind paw to move 12.6 cm sideways (the grey
cat's +x hind paw stands under the left of its body), and the bent mesh no longer showed four clear paws. **Change,
before running it again:** the targets come from the grey cat's own legs: each paw straight below where its leg joins
the body (the leg 10 to 14 cm up from the paw tip, along the surface), each pair at one front-to-back place and at least
5 cm apart sideways (each paw kept on its own leg's side), all four at the height of its lowest paw. Rest of the plan
and the keep rule unchanged.

**Second squaring try (23:05): not kept.** Paw moves up to 8.6 cm (the raised +x front paw down) and 6.9 cm; level
within 0.5 cm after it, but 2.55% of leg edges stretched over 30% (limit 2%), thin stretched spikes where the raised paw
had been, and the paw search on the squared mesh found a front pair 0.8 cm apart rather than the targets 5 cm apart.
Only 3,617 of 22,968 vertices were free to bend. **Last try, written before running it (23:10):** the same with the
whole legs and lower body free to bend (held body further than 28 cm from every paw tip; each paw held within 2.5 cm of
its tip), and one more keep condition: the paw search on the squared mesh finds each paw within 2 cm of its target. If
this fails too, the squaring is dropped and only the paw fix runs, on the current rest.

**Last squaring try (23:11): not kept, squaring dropped.** Leg edges stretched over 30%: 0.05% (fine); hind paws on
their targets (0.0 cm); but the front paws were found 4.1 and 5.0 cm off theirs: part of the -x front paw stays where it
was while its tip moves (side view, `experiments_home2/square2/home2_square.png`). As decided before the run, the grey
cat keeps its current rest pose and only the paw fix follows. The paw fix cannot bring the +x front paw down (it is
about 8 cm up and the fix is capped at 4 cm), so the grey cat is expected to fail the floating-paw part of the keep
rule; the paw fix is therefore judged per cat: Pikachu on slide (below 5%) and renders; the grey cat on the whole rule.

**Paw fix as planned (23:12, `cluster/feet_fix.py`, `experiments_feet/fix`): not kept.** The correction hit its 4 cm
cap on 53 to 77% of down frames; slide share 42% -> 36% (Pikachu recorded), 34% -> 29% (Pikachu walks), 38% -> 29% and
31% -> 23% (grey); grey floating 41% -> 28% / 47% -> 32% (+x front still 99%). Why: by height alone a paw counts as down
for seconds at a time while the whole cat turns, shuffles or lies, so one held place per stretch is wrong. Stretch was
not the problem (grey 0.80 of its own video; Pikachu's edges over 30% 13.4% -> 13.6%).

**Floor check (same run): M3's yardstick was wrong.** The page's floor (2nd percentile of the scan's heights) is
7.3 cm below the scan's surface under the paws (median of scan points within 6 cm sideways; p10 to p90 -10.1 to -2.7
cm). Against that surface the paws that are down sit 3.3 cm *inside* the floor (median; p10 10.4 cm inside, p90 3.6 cm
above). So the cat mostly sinks into the floor rather than floating; M3 as written (page floor) said it floats.

**Change, written before running it (23:15):**
- **Floor first** (`cluster/feet_floor.py`): the scan's floor level = the densest 1 cm band of heights in its bottom
  30 cm; under the cat = median height of scan points within 10 cm of that band and 8 cm sideways of the cat's lowest
  down paw (else its root). Each frame the cat is raised or lowered by the gap when its lowest point is within 8 cm of
  that surface (fading to nothing at 12 cm, so jumps onto furniture are left alone), smoothed over 0.5 s. Read-out as
  M3 but against the scan under the cat: **on the floor** if the median is within 1.5 cm and the 10th to 90th
  percentiles within 3 cm. Kept if it reads "on the floor" and renders show paws on the floor (not inside it).
- **Paw hold, second form**: held only while down and moving less than 3 cm sideways in 0.1 s (faster is a step, and
  ends the hold); same cap, easing and leg weights. M1 is unchanged, so steps faster than 3 cm still count as slides;
  also reported: the share of down steps skidding 1.5 to 3 cm (the slow skating the hold is for). Kept if that share
  falls below 5% and the stretch and renders hold; M1's own read-out is reported as it comes out.
- The grey cat's raised +x front paw stays (no fix found here); recorded as not done.

### Feet and pose check: outcome (2026-11-16, 23:20)

- **Floor fix (`cluster/feet_floor.py`, `experiments_feet/floor`): kept.** The scan's floor level came out 6.5 cm above
  the page's floor height. Against the scan under the cat, frames near the floor (46% of recorded, 54% of walk
  frames): before, the lowest point was median 2.2 cm inside the floor (recorded; walks 4.1 cm), 10th to 90th
  percentiles 7.4 cm inside to 3.9 cm above (walks 8.3 inside to 2.3 above): **floats or sinks**; after, median 0.1 cm
  inside (both), 1.8 inside to 0.8 above (walks 2.2 inside to 0.4 above): **on the floor**. Median correction 0.6 cm
  (recorded) and 1.6 cm (walks); largest 8 cm. Renders from a low camera (new address setting `elev`, degrees): before,
  the legs disappear into the floor at recorded video 8 at 46 s, walk 2 at 23 s and walk 6 at 50 s; after, the paws
  stand on the floor or rug. The grey cat shares the fix (its lowest point is put at Pikachu's).
- **Paw hold, second form (`cluster/feet_fix.py`, STEP=0.03, `experiments_feet/fix2`): not kept.** Held 80% of
  Pikachu's down paw-frames; slow skating (1.5 to 3 cm in 0.1 s) 15.6% -> 6.8% of down steps (Pikachu recorded),
  15.5% -> 7.3% (walks), grey 15.5% -> 8.5% and 15.5% -> 7.7%: above the 5% bar. M1 after: 34% and 27% (Pikachu), 29%
  and 22% (grey): still slides. Correction capped on 14% (Pikachu) to 44% (grey walks) of down frames. Stretch fine
  (grey 0.80 of its own video). Not put on the page.
- **Ship-rule renders with the floor fix** (grey cat; recorded video 18 at 90 s and 40 s, walk 3 at 5 s and 15 s): whole,
  as in version 15. No console errors (a replay with the pale real cat, and a spot walk with the grey cat).
- **On the page:** version 16 of Pikachu's Keep: the floor fix (data/feet_floor.*) for both cats and the pale real cat;
  the "At home" note says so, with the remaining slide; the grey cat's note gives the pose closeness (M4) and says its
  +x front paw stays up and why.
- **Not done:** the grey cat's raised +x front paw (about 8 cm up on every frame; its rest pose comes from a frame where
  that paw was lifted, no frame of its own video stands square, and bending the rest mesh left part of a paw behind;
  posing it square through its own skeleton would be the next thing to try). Sliding paws: about a third of down steps
  slide over 1.5 cm in 0.1 s in Pikachu's own motion (recorded and walks); holding them halves the slow skating but does
  not reach the bar. The grey cat follows Pikachu's pose loosely (M4), worst when Pikachu curls up.
- **Speed note:** on the the cluster login node (2 CPUs per user) numpy's OpenBLAS ran a 7,776 x 100 by 100 x 384 product
  in 6.4 s with its default threads and 0.1 s with `OPENBLAS_NUM_THREADS=1`; set it for CPU work there.

## Standing the grey cat square (plan, written 2026-11-17 at 01:30, before running)

The user said "Yes, try it" to posing the grey cat square through a skeleton. Findings before writing this plan
(`experiments_home2/stand`, `stand_diag*.py`):
- **Its own skeleton cannot do it alone.** One learnt bone ("L hand") carries both front legs: 98% of the weight at the
  -x front paw and 64% at the raised +x front paw (31% "L wrist", its parent). Its joints are not at elbows or wrists
  either (the "L hand" joint sits at the -x front paw). Rotating its own bones would lift or lower both front paws
  together. The hind legs do have their own bones ("R foot"/"R ankle" = -x hind, "L foot"/"L ankle" = +x hind).
- **Why the bending tries failed, probably:** the mesh is cut along its UV seams (each seam vertex is stored twice), so
  distances measured along the surface stop at the seams: every paw is "infinitely" far from every other, and parts of a
  paw on another UV piece counted as far body and were held still. That matches "part of the -x front paw stays where it
  was while its tip moves". Not rerun; the method below welds the seams first.

**Method** (`cluster/home2_stand.py`, CPU):
- Weld the seams (vertices at the same place joined) before any distance along the surface.
- **A simple skeleton per leg**, from the leg's own shape: distance along the surface from the paw tip; the leg's top
  is where the leg meets the body (the ring at that distance suddenly widens); its centre line; one middle joint at its
  strongest bend (the middle if it is straight). Each leg gets two bones (upper, lower) with weights that fade over
  1.5 cm at the joint and at the top; the body stays put.
- **Targets** as in the second squaring try: each paw straight below its leg's top; each pair at one front-to-back place
  and at least 5 cm apart sideways, each paw on its own side; all four at the height of the lowest paw.
- **Two-joint reach** (upper bone turns about the top, lower about the joint, bend kept in the leg's own plane); if a
  target is out of reach, the leg straightens toward it.
- **Moving the squared cat:** primary form, the +x front leg also gets its own bone for motion (its leg weight taken
  from the shared bones), so each front leg follows the part of Pikachu it corresponds to; second form, the 25 bones
  as now. Both then redo the correspondence and bone mixes from the squared rest, as before.

**Kept if** (the rule of the second squaring try, plus one): the squaring stretches no more than 2% of leg edges by over
30%; the paw search on the squared mesh finds each paw within 2 cm of its target; floating paws (M2) fall in recorded
and walk frames; the pose error (M4 median) grows by no more than a tenth in each (now 3.89 cm recorded, 3.32 cm
walks); the stretch read-out says "moves cleanly"; and headless renders at the ship-rule moments (recorded video 18 at
90 s and 40 s, walk 3 at 5 s and 15 s) show it whole, plus a low side view showing four paws down at a standing moment.
The primary form ships if it passes; otherwise the second form if it passes; otherwise nothing changes on the page.

### Standing the grey cat square: result (2026-11-17, 01:35; `experiments_home2/stand`, CPU)

**Not kept; nothing changes on the page.** The squaring failed its own check in both forms (they only differ after
it): 18.6% of leg edges stretched by over 30% (limit 2%), and the paw search on the squared mesh no longer found four
paws.

**Why: the grey cat's model has one front leg, not two.** With the seams welded, the two front paw tips are 10.2 cm
apart along the surface and 9.4 cm in a straight line, so the surface runs straight across from one to the other.
Seen from the front (`front_legs.png`), Pikachu's mesh has two separate front legs. The grey cat's has one column from
chest to floor, with a bump on its side 8 cm above the floor. That bump is what the paw search called the raised +x
front paw. Its canonical shape is the same (9.4 cm along the surface, 8.5 straight), so this comes from its Lab4D
model, not from the pose. That is also why one learnt bone carries both front "legs". Bending the bump down to the
floor tears the surface it grows from.

The leg-top rule (ring widening to twice its width) also misfired on the hind legs. It stopped at the -x hind leg's
hock (7.5 cm up the leg) and at the bump, so several targets were out of reach. Not pursued, since the front leg cannot
be fixed this way.

**Correction to the page:** the grey cat's note says one front paw stays up because its rest frame had it lifted. The
true cause is the model's fused front legs. The note is corrected (version 17, text only).

**Not done:** standing the grey cat on four paws needs a model of it with two front legs. A retrain on the same
133-frame video may well fuse them again; another cat whose video shows its legs apart is likelier to work. The hind
legs are separate and could be squared alone (-x hind paw 4.2 cm up, hind paws 10 cm apart front to back); not tried.

### Standing the grey cat's hind legs square (plan, written 2026-11-17 at 02:07, before running)

The user said "Yes, square them". The hind legs are separate in the mesh and have bones of their own, so they are
bent through the grey cat's **own skeleton** (`home2_stand.py`, SQUARE=hind_own; the front leg is left as it is, and
SPLIT=0):
- -x hind leg: its "R ankle" bone (upper, turning about its own joint, about 15 cm above the floor) and "R foot" bone
  (lower, turning about its joint at the hock); +x hind leg: "L ankle" and "L foot" likewise. Two-joint reach with the
  bend kept in the leg's own plane, as before. The vertices follow by the cat's own skinning (posed from its canonical
  mesh, so the weights blend the bend into the body as its model learnt). The target is corrected up to four times by
  where the paw (the mean of its 20 lowest vertices) actually lands.
- **Targets:** each hind paw straight below its leg's top (the upper bone's joint), the pair at one front-to-back
  place and at least 5 cm apart sideways, both at the height of the lowest paw. **Fallback, used only if this fails the
  squaring check:** keep the pair where it stands on average (mean sideways and front-to-back place), side by side at
  least 5 cm apart, level with the lowest paw.

**Kept if** (as before, for the hind legs): the squaring stretches no more than 2% of the moved legs' edges by over
30%; the paw search on the squared mesh finds each hind paw within 2 cm of its target; floating paws (M2) fall in
recorded and walk frames, overall and for the two hind paws; the pose error (M4 median) grows by no more than a tenth in
each (3.89 cm recorded, 3.32 cm walks now); the stretch read-out says "moves cleanly"; and headless renders at the
ship-rule moments (recorded video 18 at 90 s and 40 s, walk 3 at 5 s and 15 s) show it whole, plus a low side view at a
standing moment showing both hind paws down. If kept, it replaces the grey cat's rest on the page.

### Standing the hind legs square: result (2026-11-17, 02:10; `experiments_home2/stand/hind`, CPU)

**Not kept; the page stays as version 17.** Both target forms failed the squaring check on stretch. The paw targets
themselves were met.
- **Below the leg's top:** paws landed within 0.05 cm of their targets after two corrections, and the paw search found
  them 0.7 cm off. But 10.5% of the moved legs' edges stretched by over 30% (limit 2%). The upper bones turned 45° and
  24°, the lower 50° and 48°.
- **Fallback, in place:** 9.6% of edges over 30%. The -x hind paw landed 1.3 cm off (lower bone turned 85°); paw search
  1.8 and 0.7 cm off.
- **By eye** (`home2_stand.png`, the in-place form): the -x hind leg is dragged out sideways into a long thin shape,
  and the two hind legs merge into one blob from the side. So the picture agrees with the check.
- **Why, probably:** the learnt joints are not where a cat's joints are. The "R foot" joint sits 2.4 cm outside the leg,
  beside the paw, so turning the bone about it drags the flesh rather than bending the leg. With the cat's own
  skinning, the upper bones' weights reach well into the thigh and body. For scale: in its own video, 9.8% of all edges
  stretch by over 30%. So the model's own skinning stretches about this much anyway, but the 2% limit was set before
  the run and the shape is visibly wrong.
- **Not done:** the grey cat stays as on the page: hind paws one behind the other, the -x hind paw 4 cm up, and one
  fused front leg. A cat whose model has four separate, well-jointed legs would be the way to a square stance.

## The real cat's fur at home (plan, written 2026-11-17 at 02:40, before running; demo only)

The user picked "the real cat's fur on Pikachu at home". The At home cat is Agent-to-Sim's own Pikachu. Its 2048 px
coat was baked from its model's colour network and then brightened to the real cat's lightness (CIELAB L 74), so it has
no fur detail. The Body tab's C1 cat wears a photo coat painted from the real video frames (version 8).

**Source.** I chose to move the Body tab's photo coat onto the home mesh. Painting A2S's own home videos directly isn't
possible: those frames are not on the cluster. The README's one training-data archive (`polycam_all.zip`, 1.49 GB)
belongs to its example bunny scene: the formatting script's cat line is commented out. Even with frames, A2S's
preprocessing (masks, crops) isn't installed. Both coats show the same cat, Pikachu; the C1 photo coat comes from the
BANMo/Lab4D `cat-pikachu-0` frames.

**Method** (`cluster/home_coat.py`, CPU):
- **Pick the C1 shape.** Of the C1 cat's canonical shape and its 92 posed frames, use the one that best fits the home
  cat's rest pose after a rigid fit (one scale, rotation, shift), among those with four clear paws.
- **Find which part of the C1 cat each home vertex is.** Bend the home mesh smoothly onto that shape with paws, nose and
  tail tip pinned, as the grey cat's retarget did, with its UV seams welded first.
- **Read a colour for each home texel.** Take its point on the bent home surface and find the closest point on the C1
  surface (exact point-to-triangle, not nearest vertex, so the fur detail survives). Read the C1 photo coat there
  (bilinear).
- **Fall back to the current coat where the photo coat is weak.** Blend toward the home cat's current coat where the C1
  coat was not seen by any frame (C1's `photo_cover`, about 15% of its texels, mostly the belly). Do the same where the
  closest C1 point is over 0.5 cm away, fully by 1 cm.
- **Fill the gaps between UV pieces** as before.

**Kept if:**
1. The new coat's average lightness (CIELAB L over its texels) is within 4 of the real cat's 74.
2. **Renders, judged by eye** (current coat vs photo coat):
   - offline renders of the rest pose (front, side, top, and the head close up);
   - headless-Chrome renders at home at recorded video 18 at 90 s and 40 s, and walk 3 at 5 s and 15 s.

   The photo coat must show fur detail (stripes, face markings) that the current coat lacks. It must show no visible
   seams, no blotches, and no face features out of place: eyes and nose where the head's are, and no extra eye showing at
   the page's distance. (The C1 coat has a faint one on the side of the head.)
3. No console errors on the page.

If kept, the photo coat becomes the default at home, also on the pale real cat beside replays. A "Coat" select switches
back to the model's coat.

### The real cat's fur at home: first try (2026-11-17, 02:50; `experiments_coat/v1`, CPU, about 6 min)

**Not kept.** Lightness passes: the new coat's average L is 70.9, against the real cat's 74 (within 4). The current
coat averages 71.3 over its texels. The renders fail by eye (`home_coat_*.png`), although the fur detail is plainly
there: eyes, whiskers and tabby stripes, none of which the current coat has. The failures:
- **The face is turned toward one side.** The C1 shape chosen (frame 89; rigid fit 12 mm, the best of the 93) has its
  head turned, so the home cat's eyes and nose land on the side of its head. The front of the face is a smear.
- **Pale bluish patches with hard edges** on the belly, rear thigh, tail and the back of the neck. These are where no
  C1 frame saw the coat (15% of texels), filled from the current coat after a colour match (gains 0.81, 0.88, 1.13).
  That match turned the cream coat blue-white, and the switch between the two coats is abrupt.

Mapping detail: the closest C1 point was a median 1.4 mm away (90th percentile 3.6 mm); 0.3% of texels were over 1 cm
away.

**Change, written before running again (02:51):**
- **Head as well as body.** For each of the 40 best-fitting C1 shapes with four clear paws, fit the home cat's head
  (within 10 cm of its nose, front to back) to the C1 surface on its own, starting from the whole-body fit. Use the
  shape whose head needs the smallest extra turn; ties within 5° go to the better whole-body fit.
- **Unseen spots:** two forms, judged side by side by eye under the same rule:
  (a) C1's own colours there, as the Body tab's photo coat already has them (its model's colour network), with no
  blending toward the home coat;
  (b) the current home coat, with no colour match, fading over about 2 cm on the surface instead of switching.
- Texels over 1 cm from the C1 surface still keep the current coat (no colour match).

### The real cat's fur at home: second try (2026-11-17, 03:00; `experiments_coat/v2`)

**Not kept.**
- **The head test didn't work.** It still chose frame 89: every one of the 40 shapes needed only 7 to 35° more head
  turn, frame 89 11°, the canonical shape 8° (a tie within 5°, which goes to the better body fit). The head fit was
  poor (residual about 20 mm), so it can't tell how a rounded head is turned.
- **The face is still turned in the renders:** the nose sits right of centre, and the left eye is a smear.
- **Fallbacks:** (a), C1's own colours, has dark smudges on the thigh and tail, and its lightness is L 67.8, more than 4
  off 74. (b), the current coat fading over the surface, has no blue patches, only soft cream areas on the belly and
  thigh, and its lightness is L 72.2 (passes).
- **Check** (`experiments_coat/c1_front_head.png`): the C1 cat's own photo coat rendered from the front shows the cause.
  In frame 89 its face is turned to one side; in the canonical shape and frame 77 it looks straight ahead. The
  canonical shape fits the home cat's body as well as frame 89 does (12.1 against 12.0 mm). The C1 coat's eyes can be
  found by colour (teal texels), but I did not need them.

**Change, written before running again (03:03):** use the C1 canonical shape (straight face, equally good body fit),
and fallback (b). Everything else, including the keep rule, unchanged.

### The real cat's fur at home: third try (2026-11-17, 03:13; `experiments_coat/v3`)

**Not kept.**
- **Passes:** lightness, L 72.5 with fallback (b) (within 4). The face is now in place in the offline renders: both
  eyes, nose and whiskers face forward, with the nose slightly right of centre.
- **Fails:** headless renders at the four moments (`homecoat=photo` against `homecoat=model`; the page's own distance
  and closer). There is a sharp black blotch on the -x rear thigh, plainly visible at walk 3 at 5 s and at recorded
  video 18 at 90 s, plus brown smudges on top of the head and grey smears on the tail. The cat faces away at all four
  moments, so the face can't be judged there. The faint extra eye shows on the left cheek in the offline close-up.
- **Cause, checked:** 0.34% of the texels outside the head are very dark (L under 35), 5,093 of them in one patch on
  that thigh. On the C1 coat, two thirds of the very dark texels are spots no frame saw: its model's own colours there
  (median L 60 unseen, against 72 seen). Fallback (b)'s 2 cm fade blurs the "seen" weight outward as well as inward,
  so unseen C1 colours leak in near the edge of what was seen.

**Change, written before running again (03:15):** the fade runs only inward. The photo coat's weight is the "seen"
weight times a ramp that is 0 at the edge of the seen area and 1 about 1 cm inside it, so a spot no C1 frame saw never
takes C1's colour. Everything else unchanged. This is the last try. If the renders still show blotches, the coat is not
kept.

### The real cat's fur at home: fourth try and outcome (2026-11-17, 03:25; `experiments_coat/v4`)

**Not kept; the page stays at version 17.**
- **Passes:** lightness, now L 74.1, since the unseen spots keep the current coat in full.
- **Fails:** the black blotch on the -x rear thigh is unchanged. There are still 5,149 very dark texels in the same
  place, and it shows in the headless render at walk 3 at 5 s. So it doesn't come from spots no frame saw.
- **Cause** (`experiments_coat/c1_minus_x.png`): the C1 cat's own -x side, rendered with its photo coat, has no dark
  spot there. The home cat's outer thigh is mapped onto a dark, shadowed fold of the C1 cat, probably the gap between
  its hind leg and belly, where the two body shapes differ. Everywhere else the -x side looks convincingly like the
  real cream cat.
- As decided before this try, there are no more tries. The page code for the "Coat" select (photo coat by default,
  `homecoat=` address setting, also on the pale real cat) is saved as a patch and removed from the page. The texture
  is removed from `viewer/data`.

**What was learnt:**
- The transfer works for fur detail and lightness, and with the canonical C1 shape it puts the face in place. It fails
  where the two meshes' shapes differ enough that the closest point lands in a crease.
- A next try would treat very dark texels outside the head (L under about 35, 0.3% of texels) as unseen, so they keep
  the current coat. The open checks would then be the face at a moment where the cat faces the camera, and the faint
  extra eye at the page's distance.
- Painting A2S's own home videos would avoid the shape mismatch, but those frames are not public as far as I can tell.

## Paws that stay planted (plan, written 2026-11-17 at 05:06, before running; demo only)

The user picked "Paws that stay planted", the next of the three directions offered after version 17.

**What the earlier holds missed** (a diagnostic on `experiments_feet/check`; no fix run; `experiments_feet/skate_diag.py`):
- Pikachu's slow-skating steps are down steps (the paw down in both frames) that move 1.5 to 3 cm in 0.1 s. They are
  15.6% of recorded and 15.5% of walk down steps.
- In those steps the paw moves only 0.9 cm against the body (recorded; walks 0.6 cm; medians), but 1.5 to 3 cm across
  the room. So the body glides while the legs stay still.
- A down stretch lasts a median 1.5 s, and the paw travels a median 26 to 28 cm in it. A hold capped at 4 cm cannot
  cover that, which is why both earlier holds stopped at about 7%.
- Deciding contact by height is not the problem: in 85 to 91% of these steps the paw is over 14 cm below the body's
  centre, so the cat is standing.

**Change: short lifted steps** (`cluster/feet_step.py`, CPU). Per track and per cat, the four paws go through one state
machine, one frame (0.1 s) at a time:
- **Planted.** When a paw comes down (Pikachu's contact rule: tip at most 2 cm above Pikachu's lowest point), it is
  held where it came down, sideways. In height it is held no higher than its rest offset above the cat's lowest point,
  as before.
- **Step.** When holding it would need a correction over 4 cm (the earlier cap), the paw steps instead. It spends two
  frames in the air, 3 cm up, a third and then two thirds of the way. It then lands at its natural place plus a lead:
  its natural move over the next 3 frames, capped at 2.4 cm. The lead lets the next hold last longer.
- **Gait rule.** At most two paws are in the air at once, and never both fronts or both hinds. The paw needing the
  largest correction goes first. A paw that must wait is held at the 4 cm limit and slides with the body until it can
  step.
- **The motion's own lifts.** When the motion itself lifts a paw over 2 cm, the paw follows the motion again, easing
  out over 2 frames as before.
- **Legs.** The correction moves the paw and lower leg as before: weight 1 at the paw tip, falling to 0 at 60% of the
  leg's height.
- **Both cats.** Both use Pikachu's contacts. Each cat's steps are worked out from its own paw places. The grey cat's
  raised front leg is not touched (its model has one fused front leg).

**Measures:** the same yardsticks as the second form, after the fix, on recorded frames and walks apart, for each cat:
- slow skating: the share of down steps moving 1.5 to 3 cm in 0.1 s;
- M1: the share over 1.5 cm;
- added steps per paw per second of down time;
- the median length of a planted stretch;
- stretch: the grey cat's 99th percentile over its own video's, and Pikachu's share of edges stretched over 30%.

**Kept, judged per cat, only if all of these hold:**
1. Slow skating is below 5% of down steps, on recorded frames and on walks (the pre-set bar).
2. The median planted stretch lasts at least 0.3 s, so the paws don't tap.
3. Stretch: the grey cat at most 1.5 times its own video's 99th percentile. Pikachu's edges over 30% rise by at most
   1 percentage point (13.4% now).
4. Renders: the ship-rule moments (recorded video 18 at 90 s and 40 s, walk 3 at 5 s and 15 s) look whole. A strip of
   walk 3 from a low camera, every 0.1 s from 1 to 3 s, shows paws that stay put on the floor and then step, with no
   jitter or broken legs.
5. No console errors.

If kept, the page gets the steps (`data/feet_step.*`) as version 18, and the At home note says paws are planted and
steps were added where the motion glides. This holds for the recordings too, so the note will say the recorded motion
is changed there. If not kept, the reason is recorded here. There is one try. If its renders show a flaw that is a bug
rather than a tuning question, I get one more try, written here first.

### Paws that stay planted: result (2026-11-17, 05:10; `experiments_feet/step`, CPU, 26 s)

Read-outs applied as written (each cat; recorded frames, then walks):

| | Pikachu rec | Pikachu walks | grey rec | grey walks |
|---|---|---|---|---|
| 1. slow skating (1.5 to 3 cm), before -> after | 15.6% -> **2.6%** | 15.5% -> **1.9%** | 15.5% -> **5.7%** | 15.5% -> **3.9%** |
| M1 (over 1.5 cm), before -> after | 42% -> 6.0% | 34% -> 4.0% | 38% -> 14% | 31% -> 8.0% |
| 2. median planted stretch | **0.2 s** | 0.3 s | **0.2 s** | 0.3 s |
| added steps per paw per second down | 1.1 | 0.9 | 1.1 | 0.9 |
| planted frames held waiting at the 4 cm limit | 3.0% | 2.1% | 3.1% | 2.2% |

- **3. Stretch:** fine. The grey cat's 99th percentile is 0.71 of its own video's. Pikachu's edges over 30% go from
  11.9% to 12.1% on 80 frames where a paw carries over 2 cm.
- **Grey floating paws** (for the record): 41% -> 35% recorded, 47% -> 39% walks.
- **Why the stances are short:** 74% of planted stretches end in an added step, after a median 0.2 s (recorded) or
  0.3 s (walks). The motion's own down stretches last a median 0.7 s. The glides are fast enough that 4 cm of leg
  plus a 2.4 cm lead covers only two or three frames. This is a tuning question (the cap and the lead), not a bug.

### Paws that stay planted: renders and outcome (2026-11-17, 05:21)

**Not kept, for either cat; the page stays at version 17.**
- **Pikachu** fails rule 2 on recorded frames: the median planted stretch is 0.2 s. It passes rules 1 and 3, and on
  walks it passes 1 to 3.
- **The grey cat** fails rules 1 and 2 on recorded frames: 5.7% slow skating and 0.2 s stances.

**Renders:**
- **Ship-rule moments** (page, steps on, both cats; `experiments_feet/step/ship_steps_on.png`): whole, as in
  version 17.
- **No console errors.** Checked with walk 3 playing with the grey cat.
- **Walk 3 strip.** From the page's low camera, the room scan blocks the view or the cat faces away. So the strip is
  rendered offline: the cat alone, posed as the page does, with a fixed side camera, steps off and on
  (`cluster/feet_step_strip.py`, `LEGS=1`; `experiments_feet/step/legs_walk3_{p,g}.png`, paw tips marked).
- **Pikachu, steps on:** the steps read as steps, one paw lifted at a time while the others stay put. But on several
  frames (1.1, 1.2, 1.6, 2.2 and 3.0 s) the lower leg bends into a hook. The paw tip moves fully while the leg above it
  moves less, so the leg kinks where the weight runs out. At the page's usual distance the ship-rule renders don't show
  it.
- **Also seen on Pikachu:** the -x front paw hovers about 3 cm up and glides 2 cm per 0.1 s from 2.1 to 3.0 s, with
  steps off and on. It doesn't count as down, so neither the fix nor the measure touches it.
- **The grey cat, steps on:** thin spikes on its legs on several frames (1.0, 1.3 and 1.4 s), on top of its fused
  front leg.

**What was learnt:**
- The sliding is the body gliding over still legs. Planting paws and adding short lifted steps removes most of it:
  Pikachu's slow skating fell from 15.6% to 2.6%, and its share of down steps over 1.5 cm from 42% to 6%.
- Two things stand in the way of keeping it:
  - with a 4 cm reach, stances last only 0.2 to 0.3 s on fast glides;
  - moving the lower leg by a straight weight fall-off kinks it.
- A next try would bend each leg at its joints: two-bone reach through the skeleton's own leg bones, instead of
  sliding the leg's vertices. That gives a longer reach without kinks, and so longer stances.
- The page code (loading `feet_step`, applying it before normals, the `steps=0` and `azim=` address settings) is saved
  as a patch outside the repo. The data is removed from `viewer/data`.

## Paws that stay planted, bent at the joints (plan, written 2026-11-17 at 05:26, before running; demo only)

The user said "Yes, try it" to bending each leg at its own joints.

**Skeleton** (a diagnostic, no fix run; `experiments_feet/skel_map.py`):
- Pikachu's 25 bones are Lab4D's quadruped joints. Joint j (the page's numbering, 1 to 25) sits at the origin of bone
  j-1, 0.0 to 0.1 mm over 600 recorded frames.
- The legs are four chains of four joints each:
  - front +x: 5 -> 6 -> 7 -> 8 (bones 4 to 7);
  - front -x: 9 -> 12 (bones 8 to 11);
  - hind +x: 18 -> 21 (bones 17 to 20);
  - hind -x: 22 -> 25 (bones 21 to 24).
  The last joint of each chain is the paw.
- Segment lengths vary a little from frame to frame: 3 to 9 mm standard deviation on 4 to 11 cm segments. The skin
  around each joint follows that joint's bone.

**Change** (`cluster/feet_ik.py`, CPU). The same plant-and-step state machine as the last try, with one difference:
the leg no longer moves by sliding its surface. Each frame, every leg with a correction is solved as a two-bone chain:
- **The chain.** The top joint (shoulder or hip) stays put. The middle joint bends, keeping the leg's own bending
  direction. The third joint (wrist or hock) goes to where it was plus the paw's correction, and the paw follows it
  without turning, so it stays flat.
- **The bones.** The four bones of the leg are turned and moved with their joints:
  - the top bone turns with the upper segment;
  - the middle bone takes half of the upper segment's turn and half of the lower's;
  - the wrist or hock bone takes half of the lower segment's turn;
  - the paw bone only moves.
  The skin blends between them, as it already does.
- **Limits.** A paw is held while its correction is at most 8 cm sideways and the chain can reach it at no more than
  97% of its full length. Otherwise it steps, as before: two frames in the air, 3 cm up. It lands at its natural place
  plus a lead of its natural move over the next 3 frames, now capped at 4 cm.
- **Unchanged from the last try:** the gait rule, at most 2 cm of lowering, the motion's own lifts, and Pikachu's
  contacts.
- **The grey cat** follows Pikachu's bent bones through its usual retarget. It gets no solve of its own.
- **On the page,** each frame's paw corrections are stored, and the page solves the same chains before skinning. Its
  result is checked against the cluster's on a few frames: the joints must agree within 1 mm.

**Measures:** as in the last try. The paw places are now read from the re-posed meshes, not from targets. Stretch is
measured on 80 frames where a joint moves over 2 cm.

**Kept, judged per cat, only if all of these hold:**
1. Slow skating is below 5% of down steps, on recorded frames and on walks.
2. The median planted stretch lasts at least 0.3 s, on recorded frames and on walks.
3. Stretch: the grey cat at most 1.5 times its own video's 99th percentile. Pikachu's edges over 30% rise by at most
   1 percentage point.
4. Renders:
   - the ship-rule moments (page) look whole;
   - the walk 3 side strips (1 to 3 s, legs close up, steps off and on) show paws that stay put and then step, with no
     hooks, spikes, broken legs, or dented flanks.
5. No console errors, and the page's chains agree with the cluster's.

If kept, this is version 18. If not, the reason is recorded here. There is one try.

### Paws that stay planted, bent at the joints: result and outcome (2026-11-17, 05:31; `experiments_feet/ik`, CPU, 1 min 42 s)

**Not kept, for either cat; the page stays at version 17.** Every measured rule fails.

| | Pikachu rec | Pikachu walks | grey rec | grey walks |
|---|---|---|---|---|
| 1. slow skating, before -> after | 15.6% -> 11.6% | 15.5% -> 13.2% | 15.5% -> 13.3% | 15.5% -> 12.6% |
| M1 (over 1.5 cm), before -> after | 42% -> 29% | 34% -> 26% | 38% -> 35% | 31% -> 26% |
| 2. median planted stretch | 0.2 s | 0.2 s | 0.2 s | 0.2 s |
| added steps per paw per second down | 3.3 | 3.2 | 3.3 | 3.2 |
| planted frames held waiting (sliding) | 67% | 84% | 67% | 84% |

- **3. Stretch:** Pikachu's edges over 30% go from 12.8% to 14.4% on 80 frames where a joint moves over 2 cm, past the
  1-point limit. The grey cat moves cleanly (0.81 of its own video).
- The page's solver was written but not checked against the cluster's, and no strips were rendered: the measures
  had already settled it.

**Why** (`experiments_feet/reach_diag.py`):
- While a paw is down, the leg from its top joint to its wrist or hock is almost fully straight: 99.2%, 99.7%, 99.9%
  and 99.4% of its full length (medians; -x front, +x front, -x hind, +x hind). 87 to 100% of these frames are over
  the 97% the rule allowed.
- So the reach rule refused even the motion's own pose, and most planted paws waited and slid.
- Loosening that rule would not help. A straight leg of fixed length cannot keep its paw still while the hip glides
  along at the same height; it would have to lengthen. Bending at the joints can only shorten a leg, and here no leg
  has slack to give.
- The earlier surface-sliding fix got its 4 cm by stretching the lower leg, which is where its hooks came from.

**What was learnt:**
- The motion carries the body at a steady height on straight legs. A paw can only stay planted if either:
  - the body dips and rises as it passes over the paw, as a real cat's does; or
  - the leg lengthens a little.
- The model's own segments already vary by 3 to 9 mm (about 5%) from frame to frame. A next try could let each chain
  lengthen within that range: about 6 cm of hold on a 20 cm leg, with the stretch spread over the joints rather than
  kinked at the paw.
- The page's leg solver (`bendLegs`, `iktest=` address setting) is saved as a patch outside the repo.

## Paws that stay planted, legs that may lengthen (plan, written 2026-11-17 at 05:38, before running; demo only)

The user said "Yes, try it" to letting each leg lengthen by up to 5% while a paw is planted.

**Change** (`cluster/feet_ik.py`, with the joint-bending try's state machine and gait rule):
- **Lengthening.** When a chain's target is beyond 99.9% of its full length, both its segments may lengthen by the
  same factor, up to 1.05. The model's own segments already vary by about 5% from frame to frame. The stretch sits
  at the joints, where the skin blends between bones.
- **Holding.** A paw is held while its correction is at most 8 cm sideways and the target is within the lengthened
  chain's reach (1.05 x 99.9% of full length). Otherwise it steps, as in the last two tries.
- **Bend direction.** The middle joint now keeps the leg's own bend direction relative to the line from the top joint
  to the wrist or hock, turned with that line, so a nearly straight leg doesn't flip its knee. If the leg is straight
  to within 0.1 mm, the last try's rule is used.
- Everything else is unchanged: bones, gait, lift, lead (4 cm), lowering, contacts, and the grey cat following
  Pikachu's bones.

**Measures and keep rule:** exactly as in "bent at the joints" (rules 1 to 5, judged per cat). That includes the
check that the page's solver matches the cluster's within 1 mm, and the walk 3 side strips with no hooks, spikes,
broken legs or dented flanks. Also reported: how often a chain is lengthened, and by how much (median and 99th
percentile). If kept, this is version 18. There is one try.

### Paws that stay planted, legs that may lengthen: result (2026-11-17, 05:40; `experiments_feet/ik2`, CPU, about 2 min)

| | Pikachu rec | Pikachu walks | grey rec | grey walks |
|---|---|---|---|---|
| 1. slow skating, before -> after | 15.6% -> **5.0%** (5.02) | 15.5% -> 3.6% | 15.5% -> **12.1%** | 15.5% -> **10.3%** |
| M1 (over 1.5 cm), before -> after | 42% -> 17% | 34% -> 9.2% | 38% -> 32% | 31% -> 23% |
| 2. median planted stretch | **0.2 s** | 0.3 s | **0.2 s** | 0.3 s |
| added steps per paw per second down | 1.7 | 1.7 | 1.7 | 1.7 |
| planted frames held waiting (sliding) | 7.0% | 7.8% | 7.0% | 7.8% |
| paw off its target while down, median / p90 | 0.01 / 0.06 cm | 0.01 / 0.05 cm | 0.17 / 1.7 cm | 0.19 / 1.6 cm |

- **Lengthening:** chains are lengthened on 26% of paw-frames, by a median 2.6% and at most 5%.
- **Joint moves** where a leg is bent: median 3.9 cm, 99th percentile 8.0 cm.
- **3. Stretch:** Pikachu's edges over 30% go from 13.1% to 14.2%, which is **1.1 points**, past the 1-point limit.
  The grey cat moves cleanly (0.81 of its own video).
- **The grey cat** only follows Pikachu's bent bones through its retarget, so its paws end up 1.6 to 1.7 cm off target
  (p90) and keep skating.
- **Grey floating paws** (for the record): 41% -> 33% recorded, 47% -> 37% walks.
- **By the rule, neither cat is kept:**
  - Pikachu fails 1 on recorded frames (5.02% against below 5%), 2 on recorded frames (0.2 s) and 3 (+1.1 points);
  - the grey cat fails 1 and 2.

### Paws that stay planted, legs that may lengthen: strip and outcome (2026-11-17, 05:43)

**Not kept; the page stays at version 17.** Pikachu fails rules 1 (recorded), 2 (recorded) and 3, each narrowly or
nearly so. The grey cat fails 1 and 2.

- **Pikachu's walk 3 side strip** (`experiments_feet/ik2/legs_walk3_p.png`, legs close up, steps off and on): the legs
  now move as whole limbs. There are no hooks, no spikes and no dented flanks. Paws lift and land whole, and front
  and hind legs spread and close like a walking stance (for example 1.0, 1.7 and 2.5 s). This is the most natural of
  the three tries by eye.
- **Shoulders:** because each front leg turns about its top joint, high on the shoulder, the outline of the chest
  moves a little with it. That is expected and looks like a shoulder.
- **Not done**, since the measures had already decided: the page solver check and the page renders.
- **Comparison with the surface-sliding try:**
  - this try skates more on recorded frames (5.0% against 2.6%);
  - stances are the same (0.2 s recorded, 0.3 s walks);
  - stretch is about the same (+1.1 against +0.2 points on its own frames);
  - but the legs stay whole.
- **On walks alone,** Pikachu passes rules 1 and 2 (3.6%, 0.3 s).
- **The page code** (lengthening and bend direction included) is saved as a patch outside the repo.

### Paws that stay planted on the copy's walks: the user's decision (2026-11-17, 05:44)

The user chose "Yes, walks only": publish the lengthening try anyway, for Pikachu, on the copy's walks only. It
failed the pre-set rule (above), so this goes on the page **by the user's decision, not by the rule.**

**Scope:**
- **Applied:** Pikachu on every walk, spot and call run, and on the copy's half of the replays.
- **Not applied:**
  - the 19 recordings, which stay the real cat's estimated motion, as before;
  - the pale real cat beside a replay;
  - the grey and white cat, which follows only Pikachu's bones and gains little (10 to 12% skating).

**On walks,** Pikachu meets the skating rule (3.6%) and the stance rule (0.3 s). Stretch, measured over recorded and
walk frames together, rose 1.1 points against a 1-point limit.

**Checks before publishing** (as in rule 5):
- the page's solver agrees with the cluster's within 1 mm on the test walk frames;
- no console errors;
- the ship-rule walk moments (walk 3 at 5 s and 15 s) look whole;
- recorded video 18 at 90 s is unchanged from version 17.

### Paws that stay planted on the copy's walks: checks and version 18 (2026-11-17, 05:47)

- **Page solver against the cluster's:** the largest difference in any bent bone matrix entry is 2e-5 (metres for
  positions) over the three test walk frames, well inside 1 mm.
- **No console errors.** Checked by playing walk 3, switching to recorded video 18, then the grey cat, then back.
- **Renders** (headless page, zoom 0.4):
  - recorded video 18 at 90 s is pixel-identical with and without the fix (0 pixels differ);
  - walk 3 at 5 s and at 15 s look whole, with the legs in a new stance and a front paw mid-step;
  - the faint crease on the side at 15 s is there without the fix too.
- **Published as version 18 of Pikachu's Keep**, by the user's decision. It adds:
  - `data/feet_ik.*`: walk frames only, 1.3 MB of paw corrections plus the chains' rest joints;
  - the leg solver (`bendLegs`);
  - the `steps=0` and `iktest=` address settings;
  - a paragraph in the At home note. It says the steps missed one part of the pre-set bar (stretch) and are shown
    anyway.
