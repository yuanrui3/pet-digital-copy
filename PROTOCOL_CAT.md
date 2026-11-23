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

## A real face for the cat at home (plan, written 2026-11-17 at 05:58, before running; demo only)

The user: "the face of the cat is still somehow horrendous". The At home cat's coat comes from its model's colour
network. Its face has no eyes, nose or mouth, just a smooth pale blob
(`experiments_coat/v4/home_coat_front_head.png`, left panel).

The real-fur coat tries above already gave this cat a recognisable face from the Body tab's photo coat: two green
eyes, a pink nose and whiskers, in place with the canonical C1 shape. They were dropped only for a black blotch on the
rear thigh, far from the head.

**Change** (`cluster/home_coat.py`, new setting `HEAD_ONLY=1`, CPU): the fourth try's transfer (C1 canonical shape,
inward fade), applied to the head only.
- **Head weight.** Each texel's weight is 1 within 7 cm of the nose (front to back, in the home cat's rest pose),
  falling smoothly to 0 at 10 cm. It is also faded out below the head: full above 4 cm below the nose, nothing beyond
  7 cm below it, so the chest and front legs are untouched.
- **Everywhere else** keeps the current coat, so the thigh blotch cannot appear.
- **On the page,** the result replaces `a2s_cat_tex.jpg` for the home cat (Pikachu only). The pale real cat beside
  replays wears the same coat, as it does now.

**Kept only if:**
1. **Head renders** (offline, rest pose; front, both sides, three-quarter): two eyes, a nose and a mouth area in
   place. The face sits between the ears, not pushed to one side by more than about a quarter of the head's width.
2. **No seam:** no visible edge where the new head coat meets the old coat at the neck, in the same renders.
3. **On the page,** at two moments where the face is toward the camera (picked by turning the camera in the page):
   the face reads as a cat's face, with no third eye or smear visible at the page's normal viewing distance.
4. **Lightness:** the whole coat's mean lightness stays within 4 of the real cat's (L 74).
5. **No console errors.**

If kept, this is version 19, and the At home note says the face comes from the Body tab's photo coat. If not, the
reason is recorded here. There is one try, plus one more only if the renders show a bug.

### Result (2026-11-17, 06:08 run, checked 09:10): kept, version 19

Run: `HEAD_ONLY=1 python cluster/home_coat.py ...` into `experiments_coat/v5` (CPU). The head's photo-coat weight
covers about 4.5% of texels in full (`share_photo_full` 0.045), with 91% left fully as the old coat.

1. **Head renders** (`v5/home_coat_{front,side,other_side,three_quarter}_head.png`): two green eyes, a pink nose,
   whiskers and the white muzzle are in place. The face sits toward the head's +x side, as in the C1 coat it comes
   from. From the front it is off-centre by about 21% of the head's width, under the 25% limit. **Pass.**
2. **Seam** (`side_head`, `side_whole`): there is no hard edge at the neck. Under the chin it runs white into white. A
   darker band behind the cheek is also in the full C1 coat (the middle panels), so the transfer did not make it. The
   head reads a little more saturated than the pale body, like a soft mask. **Pass.**
3. **On the page** (local copy, headless Chrome, camera set in front of the cat by a check-only address setting that
   was not published):
   - walk 1 at 10 s, camera straight ahead of the cat;
   - walk 3 at 15 s, camera 30° round to where the head was turned.

   Both, at about 0.8 m, read as a real cat's face: two eyes, nose, muzzle, ears. Neither showed a third eye.
   **Pass.**

   Seen from the other side (walk 3 at 15 s from straight ahead of the body, with the head turned away), the smear on
   the far cheek reads as a third, squinting eye beside the two real ones. This flaw sits in the C1 photo coat itself
   (`front_head` middle panel), outside the rule's two moments. The page note says so.
4. **Lightness:** L 70.5, against the real cat's 74 (limit 4). **Pass.**
5. **Console:** no errors. The page loads `a2s_cat_tex_face.jpg`. **Pass.**

Published as **version 19**. It changes only `data/a2s_cat.json` (texture → `a2s_cat_tex_face.jpg`), the new texture
and one paragraph of the At home note. `a2s_cat_tex.jpg` stays in the artifact (unused), so reverting means switching
the json back. A possible next step would remove the far-cheek smear by fading the head weight out on the −x cheek, where
C1's coat has it, letting the old coat show there. Not tried.

## The far-cheek smudge (plan, written 2026-11-17 at 09:20, before running; demo only)

The user said "Yes, try it" to fading out the smudge on the home cat's −x cheek. It comes from the C1 photo coat and,
from that side, reads as a third, squinting eye. **Where it is:** in the home cat's rest pose, around
(−0.058, −0.148, −0.330) m. That is about 6 cm from the nearer real eye at (−0.002, −0.140, −0.344). I found it from the
v5 front render (`experiments_coat/smear_diag2.py`); a test region drawn there covers it in the front and side views.

**Change** (`cluster/home_coat.py`, new setting `SMEAR=cx,cy,cz,r_in,r_out` with `HEAD_ONLY=1`, CPU):
- Within 2 cm of that point, the photo coat is replaced by the colour of the fur around it, fading back to the photo
  coat by 3 cm.
- **The fill:** the photo coat's colours in 3 mm voxels, filled smoothly inward from the ring outside 2 cm (each inner
  voxel repeatedly averaged with its neighbours).
- So the cheek keeps the photo coat's fur tone but loses the eye-like marks, and its fur detail there.
- Everything else is as in version 19. The output goes to `experiments_coat/v6`.

**Kept only if:**
1. **Offline renders** (front, other side, three-quarter, side): no eye-like mark left on the −x cheek. The two real
   eyes, nose and muzzle unchanged. Texels more than 4 cm from the point match v5 apart from JPEG noise (mean
   difference under 1/255).
2. **No new patch:** the filled cheek does not read as a blank or blotchy spot, with no edge at the 3 cm ring, in the
   same renders.
3. **On the page:** walk 3 at 15 s, seen from straight ahead of the body, where version 19 showed the third eye: no
   third eye. The two face-on moments (walk 1 at 10 s, walk 3 at 15 s from 30° round) still read as a cat's face.
4. **Lightness:** within 4 of the real cat's (L 74).
5. **No console errors.**

If kept, this is version 20, and the At home note drops the third-eye sentence. If not, version 19 stays and the reason
is recorded here. There is one try, plus one more only if the renders show a bug.

### Result (2026-11-17, checked 09:32): kept, version 20

Run: `HEAD_ONLY=1 C1_SHAPE=canonical SMEAR=-0.058,-0.148,-0.330,0.020,0.030 python cluster/home_coat.py ...` into
`experiments_coat/v6` (CPU). The fill replaced 8,082 texels in full and faded 8,885 more; 183 inner voxels were filled
from 827 around them. Check script: `experiments_coat/smear_check.py`, picture `v6/smear_check.png` (v5 left, v6 right).

1. **Offline renders:** the eye-like slit and grey-blue streaks are gone from the −x cheek, in the front, other-side and
   close-up views. A very faint bluish tint is left where the smudge's outer edge was, in the fade ring; it is not
   eye-shaped. The real eyes, nose and muzzle are unchanged. Mean difference from v5: 11.8/255 within 2 cm,
   2.5/255 at 2-3 cm, 0.08/255 at 3-4 cm and 0.00/255 beyond 4 cm (limit 1/255). **Pass.**
2. **No new patch:** the cheek has the fur's tone, with no edge at the ring. It is smoother than the fur around it,
   having lost the photo's fur detail there. On the page at normal size this reads as a soft area. Enlarged two times,
   it is a visibly blurred patch. **Pass, narrowly.**
3. **On the page:** walk 3 at 15 s, seen from straight ahead of the body, where version 19 showed the third eye: gone.
   From the turned-away side, the squinting mark is now a faint smear. The two face-on moments (walk 1 at 10 s; walk 3
   at 15 s from 30° round) still read as a cat's face, unchanged. **Pass.**
4. **Lightness:** L 70.5, unchanged from v5 (limit: within 4 of 74). **Pass.**
5. **Console:** no errors; the page loads `a2s_cat_tex_face2.jpg`. **Pass.**

Published as **version 20**: `data/a2s_cat.json` (texture → `a2s_cat_tex_face2.jpg`), the new texture, and the At home
note's third-eye sentence replaced by one saying the smudge is smoothed over. The version 19 texture
`a2s_cat_tex_face.jpg` and the original `a2s_cat_tex.jpg` both stay in the artifact, unused.

## The face in profile (checked 2026-11-17 at 09:45; diagnosis only, nothing changed on the page)

The user: "the face still looks weird from side as profile". Page renders from the side (walk 1 at 10 s, walk 3 at
15 s, camera at ±90°) show the real-fur coat as a thin mask on the front of the face. Behind it, a dark vertical band
runs down from the ear, and then the old pale-yellow coat covers the rest of the head, the neck and the body. The Body
tab cat, wearing the same photo coat on its own shape, looks right in profile.

**Why** (`DIAG_W=1` run into `experiments_coat/v6d`, weight map in `home_coat_side_whole.png`, fourth panel):
- **The head box:** the photo coat stops at the head box, 7 to 10 cm behind the nose and 4 to 7 cm below it. Its
  straight back edge is what reads as a mask from the side.
- **Unseen spots:** behind the ear, the C1 video never saw the fur, so even inside the box those texels fall back to
  the old coat.
- **The dark band** is in the transferred C1 coat itself. It is C1's ear and the shadow behind it, landing on the home
  cat's cheek because the two heads differ in shape (12 mm rigid fit, 8° head turn).

The fourth try's whole-body coat (`v4/home_coat_side.png`) reads far more like a cat in profile. It has fur from
head to tail, with no mask edge. But unseen spots show either as pale-yellow patches (old coat) or, in the C1-colours
form, as dark blotches (the rear-thigh blotch that stopped it).

## The real fur on the whole cat at home (fifth try; plan, written 2026-11-17 at 09:50, before running; demo only)

The user chose "Whole cat" to fix the profile. The Body tab's photo coat goes on the whole At home cat, not just a box
around the face. The fourth try's transfer is reused (C1 canonical shape, inward fade), keeping the far-cheek smudge
fill. What is new is how spots with no trustworthy photo colour are handled.

**Unreliable texels:**
- unseen by C1's video;
- more than 1 cm from the C1 surface;
- or 12+ lightness points darker than the fur within 5 cm (ramping from 9 to 15). This catches the ghost-ear band on
  the cheek, the rear-thigh blotch and dark streaks.

The front 3 cm of the face is exempt from the darkness test, so the eyes, nose and mouth keep their dark marks.
`experiments_coat/band_diag.py` on the fourth try's coat flags 5.4% of texels by darkness: the ear band and the
thigh blotches on both sides, plus streaks on the back and tail.

**Change** (`cluster/home_coat.py`, new setting `FILL=1` in place of `HEAD_ONLY=1`, CPU):
- **The fill:** unreliable texels take a smooth fill from the reliable fur around them (4 mm voxels, filled from the
  nearest reliable voxel, then smoothed over the surface), never the old pale coat. Each texel blends photo and fill by
  its reliability.
- **Every texel** wears this coat. Output goes to `experiments_coat/v7`.
- **On the page** it replaces the home cat's texture. The pale real cat beside the replays wears it too, as now. The
  grey cat is untouched.

**Kept only if:**
1. **Profile** (offline side, other side and head close-ups): fur runs from the face over the head and neck to the
   body with no mask edge, and the ghost-ear band is gone. The eyes, nose and muzzle are in place as in version 20.
2. **No blotches or holes:** no dark spots or pale patches that read as marks in the whole-cat renders. Texels with
   L < 35 outside the front of the face: under 0.5% (fourth try: 4.1% of all texels).
3. **On the page:** profiles from both sides at walk 1 at 10 s and walk 3 at 15 s read as a furred cat with no mask.
   The two face-on moments still read as the cat's face, and the third eye stays gone.
4. **Lightness:** the whole coat's mean within 4 of the real cat's (L 74).
5. **No console errors.**

If kept, this is version 21, and the At home note's face paragraph becomes one about the whole coat. If not, version
20 stays and the reason is recorded here. There is one try, plus one more only if the renders show a bug.

### Result (2026-11-17, checked 10:06): passes all five rules; not yet published (the user sees it first)

Run: `FILL=1 DIAG_W=1 C1_SHAPE=canonical SMEAR=... python cluster/home_coat.py ...` into `experiments_coat/v7`
(CPU, about 8 minutes).
- **How much was filled:** 24% of texels had reliability under one half and take mostly the fill. 9.5% were flagged
  as much darker than the fur around them.
- **The fill test:** on a synthetic sphere, a hole cut in a colour gradient filled to within 0.01 of the true colour.

1. **Profile** (`v7/home_coat_side_whole.png`, `other_side_whole`, `side_head`, `other_side_head`): fur runs from
   the face over the head, neck and body to the tail on both sides, with no mask edge. The ghost-ear band is replaced
   by smooth fur colour. In the head close-up a faint outline of it and a thin line down from the ear remain. The eyes,
   nose and muzzle are as in version 20. **Pass.**
2. **No blotches or holes:** texels with L < 35 outside the front of the face fell from 4.1% to 0.01% (limit 0.5%). The
   rear-thigh blotches on both sides are gone. Filled areas (back of the hind legs, belly, behind the ears) read as
   smooth fur-coloured patches, not marks. **Pass.**
3. **On the page:** profiles from both sides at walk 1 at 10 s and walk 3 at 15 s read as a furred tan-and-cream cat,
   with no mask. Face-on (walk 1 at 10 s; walk 3 at 15 s from 30° round) the real face still shows, and there is no
   third eye. **Pass.**
4. **Lightness:** L 70.9 (limit: within 4 of 74). **Pass.**
5. **Console:** no errors; the page loads `a2s_cat_tex_fur.jpg` (local copy only). **Pass.**

The cat's overall colour changes from the old smooth pale yellow-orange to the photo coat's tan and cream fur. The user
was unsure what "Whole cat" meant, so the result is shown to them before it is published as version 21.

**The user saw the four page renders (before and after in profile, the other side, face-on) and said "Publish it".**
Published as **version 21**:
- `data/a2s_cat.json` now points to `a2s_cat_tex_fur.jpg`, the new texture;
- the At home note's face paragraph is rewritten to cover the whole coat, including the quarter filled from the fur
  around it and why.

The earlier textures (`a2s_cat_tex.jpg`, `a2s_cat_tex_face.jpg`, `a2s_cat_tex_face2.jpg`) stay in the artifact,
unused, so going back means switching the json.

## The head (free exploration, 2026-11-17, 10:20 to 11:07; nothing published)

The user: "I still feel the cat's head is a huge issue, free explore on that for some time".

**What is wrong.** `cluster/head_diag.py` (`experiments_head/diag`) compares three heads in the home rest frame: the
At home mesh, the same mesh bent onto the C1 cat (the surface the coat is transferred from, saved by
`home_coat.py SAVE_BENT=1`), and the C1 cat itself. The At home head is a tall box with a flat top, thin spike ears and
no chin. The face texture was made for the bent (C1-shaped) head, so on the box it lands off-centre with a blank cheek.
On the bent shape the same texture gives a centred, symmetric face.
- **The bend:** it turns the head 8° and shifts it 27-30 mm relative to the home head. Once that is removed, the shape
  itself changes by a median 8 mm.
- **Crinkles:** the bend is crinkled (folded small triangles). These are baked into the coat as streaks on the cheek
  and jaw.

**Tried:**
1. **Reshape** (`cluster/head_reshape.py`, `experiments_head/r3`): the head gets the bent shape. The bent head is
   moved back rigidly onto the home head, so it stays where the skeleton holds it. The change is smoothed 60 rounds
   over the welded mesh and blended into the neck (full to 8 cm behind the nose, none by 12 cm).
   - **Result:** 1,910 vertices move, median 7 mm. On the page (local copy, smooth shading, about 0.8 m), from the
     front and at 45° the face is centred with even cheeks, and the head is rounder. From the side the change is small.
   - **Unchanged:** the ears are still thin spikes and the top still a flat ridge; the home mesh has too few vertices
     there to take C1's ears. No seams, and no console errors.
   - **Page side:** a 30 KB file `data/a2s_head.*` of per-vertex offsets, loaded by a few lines in
     `viewer/index.html` (uncommitted); `head=0` in the address keeps the old head.
2. **Smoothing the bend before the coat transfer** (`home_coat.py BEND_SMOOTH`, `experiments_head/v8_s40`,
   `v8_s100`): removes the crinkle streaks but blurs the coat; the nose and mouth were lost at 100 rounds. Not used.
3. **Grafting C1's own head** (`cluster/head_graft.py`, `experiments_head/g1` to `g7`, offline only):
   - **Method:** C1's head mesh with its own photo coat is cut off behind the ears and below the chin, moved rigidly
     onto the home head, and skinned from the nearest home vertices. The home body is fitted to it: the replaced part
     is tucked inside, and the neck takes the head edge's move, fading over 4 cm.
   - **Result:** the best-looking head so far, with real ears, a round skull and the Body tab's own face; it holds up
     in posed frames.
   - **Problems:** a jagged seam behind the head, where the cut follows C1's triangles and the two coats meet, and a
     few slivers under the chin. On the page it would also need a second skinned mesh with its own texture.

**Where it stands:**
- **The reshape** is ready to check against rules and publish if wanted.
- **The graft** needs:
  - a cleaner seam: smoothing the cut edge, and fading the head's coat into the body's at the edge;
  - the page work for a second skinned mesh.

## Finishing the graft (2026-11-17, 18:10 to 19:05; published as version 22)

The user chose "Finish the graft". The work used `cluster/head_graft.py` (second round, `experiments_head/g8` to `g15`), CPU only.

**Clean cut (g8 on).** Triangles crossing the cut planes are clipped at them, so the head's open edge is a smooth line
(481 mm long, all of it on the planes). The zigzag is gone.

**Neck fitted to the head:**
1. **g8:** each neck vertex was moved to its own closest point on the Body tab cat. This crumpled the neck, with white
   spikes at the throat.
2. **g9:** the same, smoothed and blended outward. No spikes, but the strip behind the head stayed crinkled.
3. **g10, g11:** one smooth stretch, solved as a whole (sparse least squares). It keeps the stretch as smooth as possible
   while the neck near the cut sits 2 mm beneath the head.
   - **Fit:** within 0.5 mm of that depth near the cut (median).
   - **Crinkle:** none added (a vertex against its neighbours' mean: 1.33 mm before, 1.32 mm after, median).
   - **Left:** a thin crease across the throat. It is in the shape, not the coat: it stayed on a plain grey render.
   - **No better:** lower chin cuts (8.5, 9.5 and 10.5 cm below the nose; the home neck sits well inside the jowl line at
     any height), a stiffer fit, and holding only the neck points that showed outside the head (in practice all of
     them did).
4. **g13:** the crease came from dragging the head's edge under the chin down onto the neck; the jowls stand up to 16 mm
   clear of it.
   - **Change:** the edge is left hanging there (29 of 225 edge points) and laid onto the neck everywhere else.
   - **Result:** the crease is gone, and behind the head the edge sits on the neck with no step.

**Coat (g14; g15 is the same run with the cleaned-up script, byte-identical output).**
- The head's coat fades into the body's over the last 1.5 cm from its edge.
- Spots the close-up video never saw (3% of the head's texels), or much darker than the fur round them (8%), are filled
  as on the whole cat.
- The dark patch on top of the head is mostly gone: lightness under 35 outside the face falls from 3.8% to 0.7%.
- The far-cheek "third eye" does not show on this head.

**Page.**
- `viewer/index.html` loads `data/a2s_graft.*` (208 KB) and `a2s_graft_tex.jpg` (0.6 MB). The data file holds the
  neck's rest-pose offsets for 2,215 vertices, and the head: 2,776 vertices with skin weights from the nearest neck
  vertices.
- The head is a second mesh, a child of the body mesh, so it hides with it. Each frame it is skinned on the CPU from
  the same bones. The pale real cat in replays gets one too.
- The neck is only changed once the head and its coat have both loaded. `head=0` in the address keeps the old head.
  The reshape's loader (`a2s_head`) is gone.
- **Local check:** no console errors on a walk or a replay. Old and new at four views on walks 1 and 3:
  `head_compare.png` (sent to the user).
- **Page note:** a new paragraph on the head. The sentence about the face sitting to one side and the far-cheek smudge
  is removed.

**Left:**
- a small sliver at one ear base, seen from straight behind;
- the neck behind the head looks faintly faceted in the flat-shaded offline renders (smooth on the page);
- a faint lighter smudge on top of the head.

**Published (version 22, 2026-11-17, 19:10).** The user saw the comparison and said "Publish it".
- **Sent:** `index.html` plus `data/a2s_graft.json`, `a2s_graft.b64.0.txt` and `a2s_graft_tex.jpg`.
- **Checked:** the artifact's file list has all three at their local sizes. The live page could not be opened here
  (the browser pane is not signed in to claude.ai); its files are byte-identical to the local copy that was checked.

## The whole Body tab cat at home: one-day test (2026-11-17, from 19:25; checks written before building)

**Why.** The user felt the work was "stuck in a loop" and asked for an outside view (GPT 6.2, prompt only; the user
pasted the prompt and the answer themselves). Its diagnosis: the home reconstruction's crude mesh has been the visible
cat, and every fix imports pieces of the close-up (Body tab, "C1") model into it.
- **Advice:** keep the home cat only as the motion source, show the whole C1 cat moved by that motion, and compare it
  for one day against version 22, with a stopping rule.
- **The user's call:** "Run the test".
- **Method:** the grey cat's region retarget (version 15: `home2_skel.py`, `home2_retarget.py MODE=region REBASE=auto`,
  `poseGrey`), run with C1 (`cat-pikachu-0`, `fg-skel-heldout-r120`) as the target cat.

**Fixed shots (local page, same camera rules for both):**
- walk 1 (`video=19`) at t=100 from the front;
- walk 3 (`video=21`) at t=150 from the front, at +60° and -60°, and from behind (150°);
- the recording `video=17` at t=900 from the front and at 60°;
- each also with the page's own camera (no `facecam`).

**Checks (all must pass for the C1 cat to win):**
1. **Control first (diagnostic).** C1 moved by its own bones (plain skinning, as the page would do it) is compared
   with its own exported frames (Lab4D's full warp).
   - **Measure:** median vertex gap per frame.
   - **If the median is over 1 cm, or it looks wrong:** the export or skinning is suspect. One identified fix is
     allowed before going on.
2. **Shape (hard, by eye):** in every fixed shot, and on a contact sheet of 40 frames spread over the recordings and
   walks (plain grey, flat shading):
   - no merged legs, legs bent backwards, torn or collapsed body parts;
   - the head intact (round skull, ears, chin).
3. **Coat (by eye):** no new conspicuous dark or smeared patch at the page's viewing size. The C1 coat gets the same
   fill of unseen/dark spots as the version 22 head, and nothing else.
4. **Paws (measured on every page frame, both cats placed as the page places them):**
   - **Measures:**
     - paw tips into the scan floor: the median of down paws, and the share of down paws more than 2 cm into it;
     - sliding: the share of down-paw steps moving 1.5 cm or more in 0.1 s.
   - **How:** on the recordings and on the walks separately. On the walks the C1 cat follows Pikachu's bones after
     the planted-paw bending.
   - **Pass:** no worse than version 22 by more than 0.3 cm (depth) or 1.5 points (shares) on each.
5. **Speed:** in the browser pane, playing walk 3 for 10 s: the mean and 95th-percentile frame time no more than 20%
   above version 22's.
6. **Overall look:** clearly better than version 22 in at least 5 of the 7 fixed shots by my eye; then shown to the user
   side by side. The user's judgement decides.

**Stop rule.**
- **Time:** one working day for this test.
- **If the C1 cat wins:** at most two more working days, only on named failures from these checks. Each fix needs an
  identified cause and must pass the whole set again.
- **If it does not win, or the difference is unclear:** version 22 stays, and work on the home cat's looks (coat,
  head, feet tuning) stops.
- **Either way:** no more per-shot tweaks of offsets, fades or tolerances.

**Result (2026-11-17, 21:45): the C1 cat does not win; version 22 stays, and work on the home cat's looks stops.**

**Made:**
- `home2_skel.py` on C1 (`run_home2.sbatch`, which now takes `UV` and `OUT`; 3 GPU minutes, `experiments_home3/skel`).
  C1's warp is pure skinning: the 4-bone blend matches Lab4D's own warp to 0.0004 model units (median).
- `home2_retarget.py NAME=a2s_cat3 TEX=a2s_cat3_tex.jpg MODE=region REBASE=auto` (`experiments_home3/retarget`):
  - posed as its own frame 178 (four clear paws);
  - stretch 1.05× its own video;
  - its front legs are 0.27 m against Pikachu's 0.35 (hind 0.27 against 0.28).
- `cluster/cat3_coat.py`: the same fill as the version 22 head (12% of its coat unseen, 7% much darker; very dark
  fur outside the face 2.7% → 0.1%).
- On the local page copy only: a "Pikachu, close-up body (test)" choice under Cat, moved like the grey cat
  (`poseGrey`), following Pikachu's bones after the planted-paw bending on the copy's walks. The repo's page is
  unchanged.

**Checks:**
1. **Control: passed.** C1 moved by its own bones against its 92 exported frames
   (`cluster/cat3_control.py`, `experiments_home3/control`): median gap 0.37 mm (worst frame 0.83 mm, worst vertex
   11.7 mm). It looks the same.
2. **Shape: no hard failure, with a caveat.**
   - **Sheet:** `cluster/cat3_sheet.py`, 40 frames, plain grey, `experiments_home3/sheet.png`.
   - **Standing and walking:** four separate legs, nothing torn or merged.
   - **Crouching and lying:** the shoulders and front legs crease and fold (C1 only knows the poses of its 12 s
     clip). Version 22 creases in the same poses, along the neck join.
3. **Coat: passed.** No new dark or smeared patch in the fixed shots. A red spot by the tail tip from behind is the room
   scan's red bag behind the cat.
4. **Paws: failed.** `feet_check.py` + `feet_ik.py` with the C1 cat in the grey cat's place
   (`experiments_home3/check`, `ik`).
   - **Sliding on the recordings:** 37.1% of down-paw steps against version 22's 42.3% (better).
   - **Sliding on the copy's walks:** 16.8% against 9.2%. Version 22 has its own planted paws there; the C1 cat only
     follows the bent bones. That is 7.6 points worse; 1.5 were allowed.
   - **Floating:** its paws also hang over 3 cm up while Pikachu's are down: 21% of the time on the recordings and
     14% on the walks, mostly the hind paws (36-41% for one).
   - **Sinking: not measured** (corrected after the outside review). Matching lowest points (every third vertex,
     body only) makes the two deepest points equal, not the paws. When Pikachu's deepest point is its belly or tail,
     the C1 cat's lowest paw is put at that depth. Per-paw depth into the scan floor was never computed.
5. **Speed: not measured as written** (the browser pane was hidden, so the page drew no frames).
   - **Measured instead:** the per-frame posing work over 150 frames: 2.9 ms against 2.1 ms (95th percentile 4.7
     against 2.6-3.1).
   - **Scale:** about 0.8 ms more per frame, about 5% of a 60 fps frame.
6. **Overall look: not met.**
   - **Fixed shots** (`head_compare`-style sheet `compare_fixed.png`): by my eye the C1 cat is somewhat better in 4 of
     7 (a slimmer body, cleaner legs) and about the same in 3. The head is the same in both.
   - **The page's own camera** (overhead, across the room, `compare_own.png`): the two can hardly be told apart.

**By the rule written before the test:** check 4 fails and check 6 is not met, so version 22 stays.
- **What stops:** work on the home cat's looks (coat, head, feet tuning).
- **For the record:** the one clear gain of the whole C1 cat, its legs and body shape, shows mainly in close views. Its
  paws would need their own planted-paw solver, a separate project.

**Outside review of the test (2026-11-17, 22:30).** GPT 6.2 read the pets-only repository (`yuanrui3/pet-digital-copy`,
commit 298f4b2). It upholds the decision (version 22 stays) but narrows the conclusion. I checked each claim below in
the code.
- **Sliding not paired:** each cat's sliding share counts only its own down-paw steps (`feet_check.py`, M1). The
  recordings' 37.1% against 42.3% is therefore not a paired comparison, and a floating paw simply drops out of the C1
  cat's count.
- **Floating paws not shown to come from the body:** `REBASE=auto` asks for four clear paws but not level ones (that
  is `REBASE=level`). The placement is a single vertical shift, which keeps whatever height differences the posed cat
  already has. A tilted reference pose or the region mix are as likely causes as its legs.
- **Sinking:** the claim that it could not be worse was wrong (corrected above).
- **Speed:** unverified.
- **The narrowed conclusion:** the tested whole-C1 pipeline did not give enough visible benefit to justify replacing
  version 22 and building it its own planted-paw solver. Its best attainable quality, with a level reference pose and
  its own contact solver, is unknown.
- **Confirmed in the code, not fixed:**
  - the home animation is stepped at the motion's 10 samples per second, with no blending between samples
    (`viewer/index.html`, the playback loop);
  - the first 0.8 s of each walk (its recorded past) uses the walk's first predicted heading. The rollout export
    (`b1_rollout.py`) saves no past headings, and `viewer/index.html:676` substitutes `rot[0]`;
  - rare cases in the leg solver (a turn between exactly opposite directions, a perfectly straight leg) fall back
    to no turn. It is not known whether any real frame hits them.
- **Its suggestion:** one bounded day on smooth playback between samples, with gates written first.

## Smooth playback and the walk-start facing (2026-11-17, from 23:00; checks written before building)

The user: "Yes, both", that is, the outside review's one-day smooth-playback test, and the walk-start facing fix.

### A. The walk-start facing

**Found first** (`cluster/walks_past_rot.py`):
- **The past is in the recordings:** each walk's 8 recorded past frames match a stretch of the recordings to 0.7 mm
  (the worst of 132 walks).
- **The substituted heading is far off:** it differs from the recorded one by a median 32° (90th percentile 146°).
- **Same convention:** the transposed version agrees worse.
- **A jump at the hand-over:** the behaviour model's first predicted heading is often far from the real cat's at that
  moment, as above. Recorded headings in the lead-in alone would therefore swap a wrong facing for a sudden turn
  there.

**Change:**
- the 8 past frames take their recorded headings (`past_rot` in `walks.json`);
- over the first 10 predicted frames (1 s) the heading eases from the last recorded one to the predicted one (a
  rotation that shrinks smoothly to none, applied in front of the prediction; `lead_rot`);
- positions and body motion are unchanged;
- the walks' floor fix and planted-paw data are recomputed with the new headings, by the same scripts.

**Checks (all must pass):**
- **A1:** in the past frames the page's heading equals the recorded one (within 0.1°).
- **A2:** in the first 2 s after the hand-over, no frame turns more than the 99th percentile of the generated walks'
  own per-frame turning elsewhere.
- **A3:** with the recomputed data, the walks' sliding (down-paw steps moving 1.5 cm or more in 0.1 s) and floor
  gap are no worse than before: by at most 0.5 points overall, and by at most 2 points within the first 2 s of the
  walks.

### B. Smooth playback

**Change:**
- between two of the motion's samples (10 a second), every shown posed mesh (body, grafted head, the pale real cat,
  the grey cat) is blended from its two posed samples, positions and normals;
- the cat's place and turn: positions blended, turn by quaternion;
- the person marker and the skeleton lines are blended too;
- no blending across a track change, a jump of the time slider, a pause or the end of a track;
- `smooth=0` in the address keeps today's stepping.

**Fixed clips** (8 s each, chosen now):
1. walk 1 from t=9 s;
2. walk 3 from t=14 s;
3. the recording "video 18" from t=88 s;
4. the recording "video 5" from t=16 s (crouching and lying);
5. the free walk with the largest turning over 8 s (chosen by data, not by eye);
6. "Video 18 · try 1" from 0 s (the replay with the pale real cat; it includes the recorded start);
7. walk 5 from 0 s (a recorded start and its hand-over).

**Checks (all must pass):**
- **B1, exact samples:** at every sample time the shown meshes equal today's (within 0.1 mm).
- **B2, in-between frames:** halfway between samples, no vertex is more than 2 mm from the average of its two sampled
  places in the room (no swing of a planted paw, no sinking beyond the samples), over the 7 clips. The head and body
  blend with the same weights, so the seam stays within what the samples show.
- **B3, look:** blind A/B in the browser pane, the user choosing per clip without knowing which is which. The smooth
  version must be preferred in at least 5 of the 7 clips; ties do not count.
- **B4, speed:** in the visible pane, the mean and 95th-percentile frame time with blending no more than 20% above
  stepping, on clip 2.

**Stop rule:** one working day. If a check fails or the benefit is unclear, the playback change is reverted. The facing
fix is judged on its own checks. No new tuning of gait, mesh or paws.

### Results so far (2026-11-18, 00:40)

**A. The facing fix fails A2, and is not used.**
- **A1:** passes in substance. The headings are copied exactly (4 decimals). The 0.6° first measured comes from the
  arccosine formula near zero, applied to rounded matrices; the recorded headings are exact rotations.
- **A2 fails:** with the 1 s ease, the worst turn in one frame in the first 2 s after the hand-over is 22°. The walks'
  own 99th percentile later on is 16°. 3.8% of those early frames go over it, against a worst of 14° without the fix.
- **`walks.json`** is unchanged. `walks_past_rot.py` stays as the record. A longer ease would be a new attempt.

**Found while testing B: the generated walks turn suddenly every 5.6 s.**
- **Size:** 240 of the walks' 52,444 steps (0.46%) turn more than 45° in 0.1 s, up to 176°. The recordings have 2.
- **Where:** at frames 64, 120, 176, 232, ..., where the behaviour model's 5.6 s pieces meet. At these steps the cat
  really jumps in the room (walk 1, frame 119 to 120: points move up to 54 cm, 27 cm on average). The walk-start jump
  in A is the first of them.
- **Not fixed:** this is a fault of the rollout, not the page.

**B. Smooth playback: the first blending was wrong; the second passes B1 and B2.**
- **First version:** it turned the cat's frame and blended the pose inside it. At such a jump the halfway cat strayed
  252 mm from the straight blend. It was replaced before any viewing.
- **The kept version** blends the final positions in the room, as the review advised; a paw that stays put stays put.
- **Added after seeing the data, so not part of the checks written first:** a step where the cat's frame turns more
  than 45° in 0.1 s is treated as a cut and stays a step. Blending it would squash the cat through itself.
- **Over the 7 clips (557 blended steps, 4 cut steps):**
  - **B1:** at every sample the shown meshes and the cat's frame equal today's exactly (0 mm).
  - **B2:** halfway, every vertex is within 0.0001 mm of the average of its two places (float error).
  - **The largest blended move:** 441 mm in one step, in the replay clip.
- **Still to come:** B3 (the user's blind A/B) and B4 (speed in the visible pane).

**B3, the user's blind A/B (2026-11-18): fails.**
- **Picks:** the smooth version won 2 of the 7 clips (3 and 5), stepping won 1 (clip 7), and 4 were "no difference";
  5 were needed.
- **The user, afterwards:** "I feel right now the animation is still so bad that rating has no point."
- **B4:** not measured; it cannot change the outcome.

**Decision by the stop rule:** the playback change is reverted. `viewer/index.html` is back to version 22, and the
diff is kept outside the repository.

**Neither change is used; version 22 stays as published. What the two tests showed:**
- **The playback is not where the poor animation comes from;** the motion is. The home motion was estimated from
  videos filmed across the room. On the recordings, paws slide 1.5 cm or more in 0.1 s on 42% of down-paw steps,
  and the legs stay nearly straight while down ("Feet and pose check").
- **The generated walks add snaps:** the behaviour model's 5.6 s pieces join with turns of up to 176° in 0.1 s
  (above).
- **What it would take:** better animation needs a better motion source, not more work on the page or the meshes.

## Stopped (2026-11-18)

The user: "Stop here". Version 22 is the finished page. Work on the cat's looks and animation ends here, unless the user
starts a new project for a better motion source (public animal motion data, newer animal-motion models, or
re-estimating the home videos). That would start with research and a stop rule before any building.

## Restart, Phase 0: diagnostics for a better motion source (2026-11-18; tests and rules written before running)

The user asked for a roadmap ("deep analyze our roadmap to address these issues"). The report is
`reports/Cat animation improvement roadmap.md`, with its notes in `research_notes/`. Its phases are:
- Phase 0: CPU diagnostics;
- Phase 1: a seam repair inside the rollout, and a body-dip contact layer;
- Phase 2: dog-motion walking legs;
- later: retraining or physics, which need GPU time beyond the allowance.

The user: "Start Phase 0". The stop of 2026-11-18 is lifted for Phase 0 only. Phase 1 starts only on the user's word, with
its gates written first.

**Phase 0 decides three questions, plus the thresholds for judging. Nothing on the page changes.**

**P0-1, seams** (CPU, on the existing `walks.json`):
- **Measure:** at each of the walks' 5.6 s seams (and at the hand-over from the recorded past):
  - the turn in that step;
  - the cat's speed over the 0.5 s before;
  - the angle between its facing and the direction the new path takes over the next 1 s.
  Do the same for steps elsewhere in the walks and in the recordings, as baselines.
- **Read-out:** if seam snaps happen mostly when the cat was slow and the new path leaves at a large angle to its
  facing, the path stage starts paths blind to the facing (the roadmap's hypothesis).

**P0-2, does the body stage use the past facing** (GPU, a few minutes on one GPU, within the exploration allowance):
- **Measure:** for 20 seeded windows from the held-out videos, generate the body motion three ways:
  - as normally;
  - with the 8 past headings turned 90° about the vertical (positions unchanged);
  - with the past dropped.
  Then compare the first generated heading.
- **Read-out:**
  - If turning the past by 90° moves the first heading by under 20° (median), the model ignores the past facing. Only
    sampling choices, inpainting or retraining can fix the seams.
  - If it moves by over 60°, it uses it, and the seams come from the path stage.

**P0-3, reach** (CPU, recordings and walks as the page poses them, before the planted-paw bending):
- **Measure:** for every frame where a paw is down and held where it came down, how far the hip (or shoulder) must
  drop so that the leg chain reaches the held paw at 95% of its length.
- **Go-ahead for the body-dip layer:** the drop needed is at most 10% of leg length on 95% of such frames, and changes
  by at most 1 cm per 0.1 s on 95% of them, so the body need not bob.

**P0-4, rig bend** (CPU renders):
- **Test:** each of Pikachu's four leg chains is bent at its middle joint by 36° (and 45°) beyond its rest bend,
  through its own bones, in the rest pose and in 4 recorded poses.
- **Measure:** the share of the leg's mesh edges stretched by over 30%, and the narrowest leg girth against rest.
- **Pass:** no hooks or folds by eye, at most 10% of leg edges over 30% stretch, and girth at least 80% of rest.
- **If it fails:** the learnt rig blocks any fix that bends legs. Re-rigging the legs would come first, as a separate
  project.

**P0-5, error ladder** (the user's eye, local page):
- **Clips:** short clips of the page's own walks with known defects added, in a random order:
  - paw slides of 0, 4, 8, 16 and 32 mm per 0.1 s on planted paws;
  - single heading snaps of 0, 10, 20, 45 and 90°.
- **The user marks** each clip "looks fine" or "looks wrong".
- **Result:** the smallest defect marked wrong by the user sets the threshold for Phase 1's automatic guards.

**P0-6, measuring tool:** extend `feet_check.py` so that it:
- measures the motion as shown (the 10 Hz samples as the page steps them);
- pairs two candidates on the same frames;
- adds:
  - slide per stance as a share of leg length;
  - the highest planted paw speed;
  - leg extension while down;
  - turns at seams against elsewhere;
  - facing against travel direction near seams.

**Stop rule for Phase 0:** three working days. Its results are reported to the user with the Phase 1 gates written
out, and nothing is built for Phase 1 without the user's word.

**P0-1 result: seams** (`cluster/p0_seams.py`, on the page's `walks.json` and recordings).
- **Seam steps (788):**
  - 30.2% turn more than 45° in 0.1 s (median 21.6°, 99th percentile 175°).
  - Other walk steps: none over 45°, 99th percentile 6.8°.
  - Recordings: none over 45°, 99th percentile 10.6°.
- **Not the cat's speed before the seam:** the share over 45° is 26% at under 0.05 m/s, 37% at 0.05-0.15, 29% at
  0.15-0.3, and 37% above.
- **The new path's direction:** where the next window's path leaves within 45° of the facing, 12% of seams snap.
  - at 45-90°: 48%;
  - at 90-135°: 79%;
  - at 135-180°: 82%.
- **Facing against the path ahead:** at seams, a median 39° before the step and 16° after. Elsewhere it is about 16°
  throughout.
- **Read-out:** a new window's path often leaves at a large angle to the facing, and the body turns to the new path at
  once. That supports the roadmap's hypothesis, except that it does not depend on the cat being slow.
- **Not explained by the path:** two thirds of seams (525) have a path that barely moves (under 5 cm in 1 s), and they
  still turn a median 17°. That points at the body stage not holding the past facing, which P0-2 tests.

**P0-3 result: reach. Fails its go-ahead rule** (`cluster/p0_reach.py`, `experiments_p0/reach.json`; version 18's held
paws, the needed drop of the girdle, as a share of the chain's length: front 0.21 m, hind 0.17 m).

| Group | Frames | Within 10% of leg (95% needed) | Drop p50 / p95 | Smooth: change ≤ 1 cm per 0.1 s (95% needed) |
|---|---|---|---|---|
| recordings, shoulders | 5,681 | 83.1% | 1.0 / 2.5 cm | 66.8% |
| recordings, hips | 9,654 | 90.9% | 0.8 / 1.9 cm | 87.2% |
| walks, shoulders | 34,464 | 80.0% | 1.4 / 2.5 cm | 75.9% |
| walks, hips | 49,994 | 88.5% | 1.0 / 1.9 cm | 88.4% |

- **The drops are small:** about 1 cm typically and 2.5 cm at most, out of reach in under 0.4% of frames.
- **But they change fast:** a quarter to a third of frame-to-frame changes at the shoulders exceed 1 cm per 0.1 s.
- **Consequence:** a pure dip would make the body bob. A smoothed dip with the remainder taken by a little leg
  lengthening (the classic foot-skate cleanup) is a Phase 1 design choice for the user. The rule as written is not met.

**P0-4 result: rig bend. Fails** (`cluster/p0_bend.py`, `experiments_p0/bend`).
- **Numbers:** at 36° the median leg has 7.6% of its edges stretched or squeezed by over 30%, but the worst has 11.4%
  (a hind leg in a recorded pose; one front leg 10.8%). Limit 10%. The narrowest girth is a median 92% of unbent, the
  worst 79.5%; limit 80%. At 45°: 15.4% and 72%.
- **By eye** (`bend_legs_rest.png`, `bend_side.png`): at 36° the front legs tear open at the armpit and elbow (jagged
  stretched skin between leg and chest), and the torso creases. The hind legs bend smoothly.
- **Consequence, by the rule written first:** the learnt rig blocks any fix that bends the legs.
  - **Why it matters:** a body dip to 95% leg extension needs about 36° of bend at the middle joint (roadmap
    arithmetic), so both the body-dip layer and dog-motion legs depend on it.
  - **What comes first:** re-rigging the legs, as a separate project.

**P0-2 result: past facing.** `cluster/p0_past.py`, `run_p0_past.sbatch`: one A10, 7.6 min (about 0.13 GPU-hours;
the exploration total is now about 7.7 of 10). `experiments_p0/past`. 20 held-out windows × 3 seeds.
- **Turning the past facing by 90°** moves the first generated heading by a median 27.6°: the model follows about a
  fifth of the turn.
- **Dropping the past** moves it by 21.7°.
- **At the hand-over** the first heading jumps a median 23° from the last real one.
- **The new path wins:** the first heading sits a median 7.9° from the new path's direction, against 29.2° for the last
  real heading.
- **Read-out:** by the rule written first, "in between" (20-60°), close to "ignores". The facing at the start of a
  window is set mostly by the new path; the past facing only nudges it.
- **For the seam repair this means:** it has to act on the path (paths that continue the facing: best-of-N,
  inpainting of the overlap) more than on the body stage's past.

**P0-5 result: the error ladder** (the user, blind, in random order; 2 clips per level).

| Defect | 0 | small | | | large |
|---|---|---|---|---|---|
| Planted paw slide, mm per 0.1 s (0 / 4 / 8 / 16 / 32) | fine, fine | 4: wrong, fine | 8: wrong, wrong | 16: wrong, wrong | 32: wrong, wrong |
| Sudden turn of the whole walk, degrees (0 / 10 / 20 / 45 / 90) | fine, fine | 10: fine, fine | 20: fine, fine | 45: fine, fine | 90: fine, fine |

- **The turn clips did work:** in a replay the shown heading jumps the full 90° at the turn frame.
- **Thresholds for the automatic checks:**
  - paw sliding is seen from 4 to 8 mm per 0.1 s. The project's 1.5 cm line is about twice what the user notices.
  - sudden turns of up to 90° are not seen as wrong, in these clips.
- **What this says about the seams:** the snaps where the windows meet are not what makes the animation look bad to the
  user; the sliding is.

**P0-6 result: the measuring tool** (`cluster/p0_measure.py`). It measures the page as shown, at the user's thresholds,
optionally for two candidates paired on the same steps. First use: version 22 as shown, and the recordings with version
18's planted paws switched on (`REC_TOO=1`; the corrections already exist in `feet_ik.bin`, but the page uses them on the
walks only).

| Share of down-paw steps sliding more than | 4 mm | 8 mm | 15 mm | Median slide | Leg extension while down (p50) |
|---|---|---|---|---|---|
| Recordings as shown (version 22) | 82% | 60% | 42% | 11 mm | 99.3% |
| Walks as shown (version 22, planted paws) | 14% | 12% | 9.2% | 0.04 mm | 99.9% |
| Recordings with planted paws switched on | 22% | 20% | 17% | 0.05 mm | 99.7% |

### Phase 0 summary (2026-11-18)

**What the user sees:** sliding paws (from 4 to 8 mm per 0.1 s). On the page they are mostly in the 19 recordings, where
60% of planted steps slide visibly. The walks are much better at 12%. Sudden turns, even of 90°, were not seen as
wrong.

**Phase 1 options, re-ranked by these results** (none started; each needs the user's word and has its gates here):
1. **Planted paws on the recordings too.** The existing solver; its corrections are already in the page's data, so it is
   a one-line change to the page.
   - **Visible sliding on the recordings:** 60% → 20%.
   - **Trade-offs:** the legs stay straight (stretched up to 5%), and the recordings stop being "as estimated". The user
     chose walks-only in version 18.
   - **Gates:** paired measure at 8 mm on the recordings at most 25%; stretch within version 18's allowance; blind
     forced choice on recordings, at least 15 of 20 pairs for the change.
2. **Re-rigging the legs.** A separate project, and the prerequisite for any fix that bends legs (body dip, dog-motion
   legs), since P0-4 failed. Days to weeks; the risk is high.
3. **Seam repair in the rollout.** Ranked down: the user did not see the turns as wrong. P0-2 shows that any repair would
   have to act on the path.
4. **Stop at version 22.**

## Phase 1: planted paws on the recordings (2026-11-18; clips and rules fixed before building)

The user: "Paws on recordings". This is version 18's planted-paw solver, switched on for the 19 recordings as well as
the copy's walks.
- **Data:** its corrections for the recordings were computed back then (`experiments_feet/ik2/feet_ik.bin`, blob
  `rec`). The page's file holds the walks only (byte-identical to the cluster's `walk` blob), so the page file gets the
  `rec` blob added.
- **Coverage:** corrections are non-zero on 11,424 of the 11,574 recording frames.

**Gates** (from the Phase 0 summary):
1. **Paired measure:** at 8 mm per 0.1 s on the recordings, at most 25%. Measured beforehand at 20%; note this gate was
   set after that number was known, so it is not independent.
2. **Stretch:** within version 18's allowance (5%). True by construction: the same solver and data.
3. **The decider, the user's blind forced choice:** 20 pairs, the same 5 s clip with and without, sides random. The change
   passes at 15 or more of 20.

**Clips** (chosen by data with seed 1: 5 s windows where corrections are on in at least half the frames, at most 2 per
recording), as (recording index from 0, start frame):

(1, 490), (2, 380), (2, 440), (3, 10), (3, 110), (5, 310), (6, 190), (7, 260), (8, 30), (9, 10), (10, 270), (13, 840),
(14, 130), (15, 30), (15, 270), (16, 310), (17, 150), (17, 790), (18, 40), (18, 910).

**If it fails:** the page stays at version 22 and the added data is not published.

**Phase 1 outcome so far (2026-11-18): the blind test was not done; nothing published.**
- **Built, local copy only:** the page change (`recpaws`, about a dozen lines, inert without the data) and the page file
  with the recordings' corrections (`rec` blob). The page's solver matches the cluster's on 6 test frames, 3 of them
  recordings (worst difference 0.00008). No console errors.
- **The user, on opening the 20-pair test:** "sorry I am not doing this, can hardly tell, think of an unsupervised to
  way to tackle this".
- **By the gates:** the decider did not pass, so the page stays at version 22 and the added data is not published.
- **What this shows about the measure:** by `p0_measure.py`, the recordings' sliding at 8 mm falls from 60% to 20% of
  down-paw steps with the change, yet the user can hardly tell the pairs apart. So this sliding measure does not track
  what the user sees on the recordings.
  - **A likely reason:** "down" is decided by a paw's height alone, so paws that are really stepping count as
    planted.
- **Next:** a judge that needs no ratings from the user, with real cat video as the reference, to be designed and
  proposed before anything is built.

## Proposed: a judge that needs no ratings (2026-11-18; not started, waits for the user's word)

**Why.** The user will not rate pairs they can hardly tell apart, and today the sliding measure promised a large gain
that the user could not see. The judge therefore needs a reference for what real cat motion looks like, not thresholds
set by hand, and it must be checked against what the user has already said.

**Reference, already on the cluster (no download).** Lab4D's public cat-85 set, `data/cat85`:
- 77 clips of real cats (12,082 raw frames; 4,141 kept frames, whose names are their raw frame numbers, so times are
  known);
- 29 of the clips are this same cat (3,979 raw frames);
- for every kept frame the archive holds the cat's outline (mask), the image motion to the next frames (flow), depth,
  and an estimated camera direction relative to the cat.
- **Not usable for this:** the user's curated clips are sounds, grooming and resting, with no walking.

**Method.**
1. **Our cat in the same form.** Outline sequences of the animated cat are rendered from the real clips' own camera
   directions, at the same time spacing. This is exact and needs no estimator.
2. **Named gait numbers,** measured identically on real and ours:
   - body lengths travelled per leg cycle (from the outline's leg band and the ground's image motion);
   - leg swing amplitude;
   - bend of the leg outline;
   - body bob.
   Where ours falls outside the real cats' range, that is a named defect.
3. **A blind test by machine:** a small classifier is trained to tell real outline-motion clips from ours and scored
   on held-out clips (50% means it cannot tell them apart). Its errors and attributions show when in the cycle, and
   where on the body, ours gives itself away.

**Trust checks (all must pass before the judge decides anything):**
- **T1, real against real:** two halves of the real clips, split by clip: at most 60%.
- **T2, same motion, model against video:** the two clips for which both the video and our own reconstruction exist
  (`cat-pikachu-0`; `cat-85-0068`, the grey cat): the reconstruction's outlines must score clearly closer to real than
  the page's walks and recordings do. This separates "the motion is wrong" from "it is a 3D model".
- **T3, agreement with the user's known judgements:**
  - walks with added paw slides of 16 and 32 mm score worse than without;
  - version 18's planted walks score better than the un-planted ones;
  - planted paws on the recordings change the score only a little.
- **If a check fails:** the judge is not used, and that is reported.

**Use.** Each candidate fix is scored against real cats on the same clips (paired, with a bootstrap noise band). The
user is shown one before/after only when the judge reports a large change.

**Limits.**
- about 7 minutes of real footage;
- outlines alone miss some paw sliding, so the ground's image motion is used with them;
- the judge names defects but does not fix them, and the front legs still cannot bend (P0-4).

**Effort and stop rule.** CPU, no download, about 2 working days. If the trust checks fail, the work stops and the page
stays at version 22.

## Building the judge (2026-11-18; the user: "Build it"; details fixed before any result)

**What is measured, the same way on real clips and on ours.** A "pair" is two consecutive frames. Everything is taken from
the cat's outline pixels and their image motion, in units of the cat's size (the square root of its outline area), as
ratios to how far the body travelled in that pair. Ratios do not depend on the clips' frame rates, which are unknown.
- **Travel:** the body's mean image motion minus the ground's (the background just under the feet).
- **Paw slip ratio:** in the lowest strip of the outline (bottom 8% of its height), the slowest tenth of the motion
  relative to the ground, divided by the travel. A cat with a planted paw gives about 0; a gliding cat about 1.
- **Leg motion ratio:** the mean motion of the lower 40% of the outline relative to the body, divided by the travel.
- **Torso motion ratio:** the same for the upper half.
- **Stride:** the leg motion's cycle (autocorrelation of the fore and hind halves' motion along the travel) times the
  travel, in sizes per cycle; and the swing amplitude.
- **Windows:** about 12 pairs; kept only if the cat is whole in the frame, longer than tall (ratio at least 1.3),
  travelling within 30° of the image's horizontal, at 0.3 sizes per second or more (30 frames per second assumed for
  this threshold only).

**Ours.** The page's cat (version 22's shape) is drawn as points and their exact image motion.
- **Camera:** a still, level camera beside the path, 55° to 125° from the travel direction and 5° to 30° above it,
  drawn at random with a fixed seed.
- **Pair spacing:** chosen so the travel per pair matches the real clips' typical travel per pair.
- **Blur:** real video motion is estimated and blurred at fine scale, so ours is blurred by 3% of the cat's size. This
  is set from the estimator's resolution and not tuned; 0% and 6% are reported beside it.

**Changes from the proposal, made now:**
- **Cameras:** side views are selected from the image itself and ours are drawn from a fixed range. The set's camera
  estimates are coarse.
- **Check T2:** it compares the leg and torso measures, real video against our own reconstruction of the same clip, on
  the two clips that have both. The slip ratio cannot be checked that way, because the reconstructions have no ground.
  Its check is physical: real walking cats always have a planted paw, so whatever the real clips measure above 0 is
  the instrument's floor.

**Blind test by machine.** Logistic regression (and a small forest, for comparison) on the window measures, motion
ratios only, no shape measures. Folds are grouped by clip, and the score is the held-out balanced accuracy.

**Window measures and the checks' numbers, fixed before any result (2026-11-18).**
- **Window measures** (12 pairs; sums over the window, divided by the summed travel):
  - slip ratio (10th and 25th percentile versions);
  - leg motion ratio;
  - torso motion ratio;
  - the fore and hind leg parts' swing (standard deviation of their motion along the travel, over the mean travel);
  - their mean;
  - the correlation between fore and hind.
- **Score of a candidate:** the classifier (trained on real windows against version 22 as shown) gives each of its
  windows a probability of being real. Candidates measured on the same windows and cameras are compared by the mean
  paired difference, with a 90% interval from resampling tracks.
- **T1 passes** if real against real (clips split at random into two halves, 20 splits) averages at most 60% balanced
  accuracy.
- **T3 passes if all three hold:**
  - (a) walks with 16 and 32 mm of added slide both score lower than the walks as shown, the interval excluding 0;
  - (b) the walks as shown (planted) score higher than the walks un-planted, the interval excluding 0;
  - (c) planting the paws on the recordings changes the score by less than half of what planting changes on the walks
    in (b), or by less than 0.05.

**Real clips measured; the window definition loosened before any comparison with ours (2026-11-18).**
- **Measured:** `cluster/judge_real.py` (`experiments_judge/real_pairs.npz`): 2,680 of 4,064 pairs, with the ground
  found under the feet in all of them.
- **Too few windows as first defined:** 12 pairs and at least 1.3 times longer than tall gives 40 windows from 19 clips.
  The limit is runs of 12 measurable pairs in a row: only 98 exist with no other filter.
- **Change, made on these counts alone:** windows of 8 pairs, a step of 4, and at least 1.15 times longer than tall. That
  gives 93 windows from 28 clips. The other limits stay (travel within 30° of horizontal, 0.01 sizes per raw frame).
- **First look at real cats** (the 40 strict windows, medians):
  - slip ratio 0.12 (quartiles 0.08 to 0.29): the slowest part of the paw strip moves at about a tenth of the body's
    speed. This is the instrument's floor for a truly planted paw;
  - leg motion ratio 0.59;
  - torso motion ratio 0.45;
  - travel per pair 0.08 sizes.
- **Ours** is drawn with `TARGET=0.08` (travel per pair) and 8 pairs per window.

**Check T2, made exact before running (2026-11-18).**
- **Clips:** the two that have both a video and our own reconstruction of it: `cat-pikachu-0` (the Body tab cat) and
  `cat-85-0068` (the grey cat).
- **Reconstruction side:** the reconstruction's posed frames (body frame, articulation only) are drawn from the video's
  estimated view direction. They are measured like ours, with the body's travel taken from the video (the
  reconstruction has no ground): the ground's motion is set so that the outline's travel equals the video's over the
  same raw frames. A paw that the reconstruction keeps planted then measures a slip near 0.
- **Windows:** common spans of 24 raw frames, a step of 12, where every video pair in the span was measured and the cat
  travels at 0.01 sizes per raw frame or more.
- **T2 passes if, on both clips:**
  - (i) the reconstruction's median leg motion ratio is within a factor of 1.5 of the video's, and its median slip
    ratio is within 0.15 of the video's;
  - (ii) the classifier (trained without that clip) gives the reconstruction's windows a mean probability of being real
    that is nearer to the same clip's video windows than to version 22's.

**Check T2, first version: not established (2026-11-18).** `cluster/judge_recon.py`.
- **Too few windows:** the grey cat's clip gave no common window and the Body tab cat's clip gave 3. Its video is a
  close-up, and the outline touches the frame's edge in most pairs.
- **In those 3 windows:** the leg and swing measures follow the video closely (correlations 0.97 to 1.0, at about two
  thirds of the size), but the slip ratio does not (0.46 against the video's 0.05).
- **A flaw in the check:** the reconstruction was drawn in its body frame from the set's coarse view estimate, with
  the travel borrowed from the video. That is not the image the model actually reproduces.

**Check T2, corrected (rule unchanged; written before running).**
- **Reconstruction side:** each reconstruction is posed at every kept frame of its own video in the model's own
  optimised camera, with that frame's intrinsics (`cluster/judge_cam_export.py`, a few GPU minutes inside
  `run_judge.sbatch`), and drawn into the video's own pixels.
- **Pairs:** the video's own (kept frame to next kept frame).
- **Ground:** the video's measured ground motion for that pair.
- **Frame edge:** the outline is cut at the frame's edge on both sides alike, so pairs where the cat touches the edge
  count.
- **Windows:** 8 pairs, a step of 4, where both sides are measured and the cat travels at 0.01 sizes per raw frame or
  more.
- **Pass rule:** (i) and (ii) as written above.

**Where the drawing runs:** the login node was too slow for drawing ours (125 windows in 25 minutes), so the six
versions, four blur variants and the two camera exports run as one job on a compute node (`run_judge.sbatch`, one A10,
12 cores).

## The judge's results (2026-11-18): a sound ruler, but not a stand-in for the user

**Runs.**
- **Job:** `run_judge.sbatch`, job 12420621: one A10 for 7 min 45 s, about 0.13 GPU-hours (the exploration total is now
  about 7.85 of 10).
- **Ours:** `judge_ours.py`: six versions and four blur variants, 220 windows each on the same windows and cameras.
- **Scoring:** `judge_score.py` (`experiments_judge/score.json`).
- **Real:** 93 windows from 28 clips.
- **Speeds match:** real 0.023 sizes per raw frame, ours 0.026.

**The gait numbers (medians; real quartiles in brackets).**

| Measure | Real cats | Version 22 as shown | Walks, planted (shown) | Walks, un-planted | Recordings (shown) | Recordings, planted |
|---|---|---|---|---|---|---|
| Slip ratio (slowest paw's speed over the body's) | 0.16 (0.09 to 0.36) | 0.44 | 0.36 | 0.56 | 0.54 | 0.42 |
| Leg motion ratio | 0.60 (0.49 to 0.78) | 0.28 | 0.34 | 0.15 | 0.21 | 0.33 |
| Torso motion ratio | 0.53 (0.31 to 0.79) | 0.25 | 0.26 | 0.16 | 0.22 | 0.29 |
| Fore-leg swing | 0.26 | 0.13 | 0.18 | 0.05 | 0.08 | 0.16 |
| Probability of being real (classifier) | 0.85 (held out) | 0.11 | 0.15 | 0.02 | 0.05 | 0.14 |

**The blind test by machine:** real against version 22: balanced accuracy 93%, AUC 0.97 (recordings 94%, walks 91%). It
calls 94% of real windows real and 7% of ours real.

**Trust checks.**
- **T1 passes:** real against real 47% on average over 20 splits (highest 63%); 60% was the limit.
- **T2, corrected: passes on one clip, fails on the other.**
  - **Reconstruction outlines:** drawn in the models' own cameras, they match the videos' in size (0.98 and 0.95).
  - **Body tab cat's clip (9 windows): passes.** Slip 0.10 against the video's 0.08; leg ratio 0.92 of the video's;
    correlations across windows 0.87 to 0.99 on every measure; probability of being real 0.48 against the video's 0.62
    and version 22's 0.11.
  - **Grey cat's clip (7 windows): fails the slip part.** Slip 0.67 against the video's 0.41, a difference of 0.26 where
    0.15 was allowed. The leg ratio is 1.09 and the score part passes (0.58, video 0.85). This reconstruction is the
    known weaker one (one fused front leg, outline overlap 0.88), so the difference may be the model's own sliding
    and not the instrument, but the rule asked for both clips.
- **T3: (a) and (b) pass, (c) fails.**
  - **(a) passes:** added slide of 16 mm lowers the score by 0.059 [0.017, 0.111], and of 32 mm by 0.128 [0.085, 0.182].
  - **(b) passes:** planted walks score 0.140 higher than un-planted [0.095, 0.192]. With no blur it is 0.159, and with
    6% blur 0.131.
  - **(c) fails:** planted paws on the recordings raise the score by 0.105 [0.076, 0.130]. That is three quarters of the
    walks' gain, where under half (0.070) or under 0.05 was required. The user could hardly tell those pairs apart.

**Verdict by the rules written first:** the judge is not used to decide for the user. Check (c) and one clip of T2 failed.

**What the results still show** (from an instrument that passed T1, T2 on the well-reconstructed clip, and (a) and (b)):
- **Every version is far from real cats, and in the same ways.** Their legs move about half as much per distance
  travelled (0.28 against 0.60). The slowest paw moves at 0.44 of the body's speed against 0.16. The torso is about half
  as lively.
- **Planted paws help by this ruler, on walks and recordings alike, but only a little on its scale:** from 0.02-0.05 to
  0.14-0.15, where real cats score 0.85.
- **Why the user can hardly tell (consistent with their own words):** the versions being compared all sit at the
  "clearly not real" end.
- **To look different to a person,** a change would have to move these numbers most of the way to real: legs that move
  twice as much and a livelier body. Phase 0 showed the present rig cannot bend the front legs for that.

**Stop rule applied:** the trust checks did not all pass, so this work stops and the page stays at version 22. The
recordings' planted paws are not published.

## Stopped again (2026-11-18)

The user: "Stop at version 22". Version 22 is the finished page.
- **Not published, and taken out of the page file:** planted paws on the recordings. The page change is kept as
  `recpaws_page.patch` in the sessions' shared notes folder; the recordings' corrections are the `rec` blob of
  `experiments_feet/ik2/feet_ik.bin` on the cluster.
- **Kept for a later leg rebuild, should the user ever start one:** the ruler (`cluster/judge_*.py`,
  `run_judge.sbatch`, `experiments_judge/` on the cluster) and the Phase 0 tests (`cluster/p0_*.py`). The ruler says how
  far a version is from real cats; it is not to be used in place of the user's judgement (check (c) failed).
- **What a visible improvement would need:** legs that move about twice as much per distance and a livelier body. That
  means re-rigged legs with new walking motion, a project of weeks with real risk.

## Better data: screening the close-up clips (2026-11-18; no rebuild started)

The user asked "should we look for better data", then "Screen the clips". The idea: the ruler showed that the Body tab
cat, rebuilt from a close-up clip, moves like its video, while everything from the distant home videos does not. The
cat-85 set on the cluster has more close-ups that an earlier note here called "Pikachu itself" (folders 0055 to 0084).

**That note was wrong, and so was what I first told the user** ("29 clips of this same cat, about 38 s of clean side-view
walking in 20 of them"). Looked at properly (`cluster/clip_screen.py`, picture sheets `experiments_judge/screen/`):
- **Other cats:** 0068 is the grey cat (our own second cat) and 0069 is a white long-haired cat.
- **Not walking:** the ruler's "side-view walking" count included a cat rolling on a carpet seen from above (0078 to
  0080, 18 of the ruler's 93 real windows).
  - **Effect on the ruler:** without those clips the real cats' medians are slip 0.12, leg 0.60 and torso 0.45, and
    the machine blind test is 94% (AUC 0.97). The ruler's conclusions do not change.

**Which clips are this cat** (`cluster/clip_identity.py`, `run_clip_identity.sbatch`; two jobs, 5.5 GPU minutes).
- **The cat test's identity check, as it was, does not carry across clips.** With the Body tab clip as the only
  template it accepts only clips that look like that clip (0055, and partly 0062 to 0064). It rejects clips filmed in
  the same home that are plainly the same cat.
- **Clip-to-clip score instead:** each frame's three nearest frames among the known clips of this cat, minus the same
  among known other cats; the median over the clip's frames.
  - **Known this cat** (0055 to 0065 and the Body tab clip, same home, left out one at a time): all 12 above 0
    (+0.04 to +0.36).
  - **Known other cats** (50 clips): median -0.06, 5 above 0, highest +0.039.
- **Unknown clips:**

| Clip | Score | Reading |
|---|---|---|
| 0081, 0070 | +0.10, +0.04 | this cat (the same table as the known clips) |
| 0066, 0071, 0073 | +0.05, +0.05, +0.04 | likely this cat (above every other-cat clip) |
| 0072, 0083, 0077, 0078 | +0.01, +0.03, +0.01, 0.00 | unclear (0072 is the same scene as 0071 and 0073) |
| 0074, 0075 | -0.06, -0.05 | scored like other cats; a grey-tan cat in the same room as 0071 to 0073 |
| 0067, 0079, 0080, 0082, 0084 | -0.02 to -0.05 | scored like other cats, or unclear |

- **By eye:** the cat of 0071 to 0073 is ginger, and the Body tab cat is pale gold. But in 0061 and 0056, filmed in the
  known home, this cat also looks ginger in warm light, so colour alone does not rule 0071 to 0073 out.

**Side-view walking on the floor, whole cat in frame** (by eye; seconds from the ruler's windows):

| Clip | Length | Side walking | Walks towards | Cat size | This cat? |
|---|---|---|---|---|---|
| 0071 | 3.2 s | 2.9 s | left | 326 px | likely |
| 0072 | 4.8 s | 4.1 s | right | 315 px | unclear (same scene) |
| 0073 | 4.2 s | 2.9 s | left | 557 px | likely |
| 0066 | 3.0 s | 1.3 s | left | 511 px | likely |
| 0074 | 10.5 s | 7.1 s | right | 482 px | scored as another cat |
| 0075 | 6.5 s | 6.1 s | left | 369 px | scored as another cat |
| 0055 | 4.0 s | 0.4 s (walks, then turns to the camera) | right | 630 px | yes |
| 0060 | 2.7 s | 1.1 s (on a table) | left | 516 px | yes |

- **Total for this cat:** about 10 to 12 s of clean side walking that is confirmed or likely this cat, in both
  directions. The two best walking clips (0074 and 0075, 13 s more) may be another cat.
- **Other views of this cat:** the remaining known clips (0056 to 0065) are front views, on a table, or sitting.
  They are useful for shape, less so for gait.

**What a rebuild would cost** (Lab4D's own recipe for several clips of one cat; the archive already holds every
preprocessed input, so there is no download):
- **Per round:** about 165 s on one A10, whatever the number of clips (the Body tab cat: 120 rounds, about 5.5
  GPU-hours; the grey cat: 60 rounds, 2.7).
- **For about 10 clips:** 120 to 240 rounds, so 5.5 to 11 GPU-hours, plus a 4-round trial (0.2) and exports (0.3).
- **Budget:** about 7.95 of the 10 exploration hours are used, so a rebuild needs the user's approval.
- **After it,** the walking pieces must still be joined to follow the home paths, and checked with the ruler.

## The rebuild from this cat's close-up clips (2026-11-18; plan and pass rules written before any run)

The user saw the screening and chose "Approve the rebuild" (offered at about 6 to 12 GPU-hours). This GPU time is
approved separately from the 10 exploration hours (about 7.95 used). **Hard cap: 12 GPU-hours** for the trial,
training, exports and checks together. If anything would exceed it, stop and ask.

**Clips (18): only those certain or likely to be this cat.**
- **The Body tab clip:** `cat-pikachu-0-0000`.
- **Certain, same home:** `cat-85-0055` to `0065`, `0070`, `0081`.
- **Likely:** `0066`, `0071`, `0072`, `0073` (0072 is the same scene as 0071 and 0073).
- **Left out:** 0074 and 0075 (scored as another cat), the unclear clips, and all other cats.

**Training:** Lab4D's recipe for several clips of one cat, with the settings that built the Body tab cat (`fg-skel`,
`--fg_motion skel-quad`, 256 px crops, one A10); every clip is treated as the same cat.
- **Trial first:** 4 rounds, to time a round and to see that the 18 clips load.
- **Full run:** 160 rounds if a round takes 200 s or less (about 7.3 hours at 165 s); if slower, 120 rounds. If even 120
  rounds would pass 9 hours, stop and ask.

**Pass rules for the rebuilt cat** (all needed before any work on walks):
- **R1, outline:** drawn in the model's own cameras, its outline overlaps each clip's (intersection over union). The
  median is at least 0.85 on each of the walking clips 0071, 0072, 0073 and 0066, and on the Body tab clip (the
  single-clip Body tab cat reached 0.945, the grey cat 0.88).
- **R2, legs by eye:** four separate legs, nothing fused or torn, on a sheet of frames from the walking clips, drawn
  plain grey.
- **R3, the ruler on its own videos:** on the walking clips together, the reconstruction's median leg motion ratio is
  within a factor of 1.5 of the video's and its median slip ratio within 0.15 of the video's (check T2's rule (i)).
- **R4, the ruler's score:** the classifier's mean probability of being real, for the reconstructed walking clips, is at
  least 0.40 (the Body tab cat's reconstruction scored 0.48; version 22 scores 0.11).

**If any of R1 to R4 fails:** the rebuild is not used, the page stays at version 22, and the result is reported.
**If all pass:** the next step is proposed to the user before it starts: joining the walking pieces so the cat follows
the home paths.

**Found while setting up (2026-11-18), before any run: the walking clips are thin in time.**
- **Frames kept by the set's own preprocessing:** it keeps only frames with enough image motion. For the walking clips
  that leaves 14 (0071), 15 (0072) and 23 (0073) frames, about 3 to 6 a second; 0066 has 62. The 18 clips have 943 kept
  frames in all.
- **Precedent:** the Body tab clip was as sparse (76 of 366 raw frames, about 6 a second) and its rebuilt motion measured
  close to its video.
- **Decision:** the rebuild goes ahead as approved.
- **The remedy, if the result is promising but too coarse:** re-run Lab4D's preprocessing on those clips' raw frames
  (they are in the archive). That needs its tools installed first, which means downloads to approve.

**Trial (2026-11-18, 03:00): passed.** Job 12422642, 4 rounds on one A10: the 18 clips load, and a round takes 132 to
143 s. The whole job took 13 minutes (0.22 GPU-hours).
- **Full run, as planned:** 160 rounds, expected about 6.1 hours (`run_pika18.sbatch 160`, limit 8 hours).
- **Export test:** the export and checks (`run_pika18_export.sbatch`) are first run on the 4-round trial model, to find
  mistakes in them before the full model is ready. Its numbers mean nothing.
- **Expected GPU total:** about 6.7 hours of the 12 allowed.

**Running (2026-11-18, from 03:15):** the full training, job 12422934 (160 rounds; rounds so far 134 to 142 s).
- **Export test on the trial model:** job 12422935, 6.6 minutes; the export and all four checks ran end to end.
  - **Its numbers are a 4-round model's and mean nothing** (outline overlap about 0.6, legs barely moving).
  - **A useful by-product:** the walking clips' own videos score well on the ruler (probability of being real 0.59 to
    0.94, slip 0.09 to 0.13).
  - **Windows available for R3 and R4:** 16 in the four walking clips (0066: 9, 0071: 2, 0072: 2, 0073: 3).
- **Queued:** the same export and checks for the full model, to start when the training ends
  (`run_pika18_export.sbatch 160`). Outputs go to `experiments_pika18/fg-skel-r160/`.
- **GPU so far on the rebuild:** 0.33 hours.

**Result (2026-11-18, 09:24): failed R1, R2 and R4. The rebuild is not used; the page stays at version 22.**
Training job 12422934 ran 160 rounds in 6 h 03 min; export and checks job 12423059 took 8 minutes. GPU used on the
rebuild in all: about 6.5 hours of the 12 allowed (trial 0.22, export test 0.11, training 6.05, export 0.14).
- **R1, outline: FAIL, narrowly.** Medians: 0071 0.863, 0072 **0.847**, 0073 0.875, 0066 0.874, the Body tab clip
  **0.824** (needed 0.85 on each). The other 13 clips: 0.850 to 0.888.
- **R2, legs by eye: FAIL.** On the walking clips' sheets the cat has a cat's outline from the camera, with short stub
  legs that often read as two, not four. On the Body tab clip the same model's cat is crumpled into a folded shape
  (0.26 across the picture in a side-view frame, against 0.42 on 0073), though its outline from the camera still
  overlaps 0.82. *(Corrected the same day: this first said the body was "a flat slab, a cardboard cut-out" seen from
  other directions. Measured, it is not; see the correction below.)*
- **R3, the ruler on its own videos: pass, narrowly.** Walking clips pooled (16 windows): leg motion 0.74 of the
  video's (allowed 0.67 to 1.5); slip +0.140 over the video's (allowed 0.15).
- **R4, the ruler's score: FAIL.** Mean probability of being real for the reconstructed walking clips 0.16 (needed
  0.40); their own videos 0.68; version 22 0.10. Per clip: 0066 0.19, 0071 0.13, 0072 0.03, 0073 0.18; the Body tab
  clip 0.14 (its single-clip model scored 0.48 on the same clip).
- **Root cause (first written; withdrawn, see the correction below):** "the clips are nearly all side views from
  about the same height, so nothing pins the cat's width; the model fits the outlines with a flat body and stiff stub
  legs". Both halves were wrong.
- **What the checks do show:** stub legs, slip and leg motion near version 22's, and joint training made the Body tab
  clip worse than its own single-clip model. The thin frame counts (14 to 23 kept frames on the walking clips) may add
  to it, but the Body tab clip had 76 and still collapsed, so denser frames alone are not shown to be the fix.
- **Stop rule honoured:** no repair runs. Not tried: re-processing the raw frames at full rate (needs tool installs
  and downloads to approve); training the walking clips without the Body tab clip.
- **The user chose "Stop here" (2026-11-18).** The full-frame-rate re-processing is not done. No cluster jobs are left
  running; the page stays at version 22.

**Correction (2026-11-18, measured after the result was reported): the rebuilt cat is not flat, and the clips are not
all side views.** Found while preparing the theory report (`research_notes/Cat animation bottleneck theory/
coordinator_measurements_and_corrections.md` has the tables).
- **Width, measured:** the rebuild's rest mesh is 0.096 wide, like the Body tab cat (0.098) and the grey cat (0.104);
  mid-body width/height 0.48 against 0.54, 0.56 and the home cat's 0.39. Posed frames keep it: thinnest/longest ratio
  0.23 to 0.32 on all 17 cat-85 clips (rest 0.24). The "cardboard cut-out" came from my reading orthographic renders
  of a long cat with its tail held out; seen from above or end-on a real cat is narrow too.
- **The Body tab clip is folded, measured:** posed ratio 0.46 with the longest spread 0.0715 against 0.095 to 0.13 on
  every other clip.
- **View angles** (clip screening, each clip's preprocessing camera in the cat's frame): across the 17 cat-85 clips
  the median azimuth spans most of the circle and the elevation 9 to 35 degrees. Only within each walking clip is the
  view nearly fixed (p10 to p90: 0071 11, 0072 16, 0073 10 degrees).
- **The failure stands** (R1 narrow, R2 stub legs and the folded Body tab clip, R4 0.16); its cause is not known.

## A built-in walking cycle on the copy's walks (2026-11-18; checks and stop rule written before building)

The user asked for a deep dive into the theory behind the bottlenecks (`reports/Cat animation bottleneck theory.md`),
then asked what the next step should be. Recommended: one bounded try at replacing the legs on the copy's walks with a
built-in cat walking cycle, built from published cat gait numbers, keeping the behaviour model's paths and body. The
user said "Try it". If it is not clearly better to the user in one before/after, version 22 is final.

**Why:** the report found that the binding limit is the source leg motion (legs move about half as much as a real
cat's, stay about 99% straight while a paw is down), which no fix layered on top of it can supply. Every route to
better legs substitutes an assumption; the built-in cycle is the only one that needs no new data, no generative model,
no uncertain licence and no GPU.

**Facts read off the page's data first** (`cluster/gait_probe.py`, all 132 copy runs, 52,576 frames):
- **Legs** (top to middle to wrist or hock, then to the paw): front about 0.10 + 0.11 m, then 0.06 m; hind about
  0.10 + 0.075 m, then 0.04 m. Extension while walking: median 99.0 to 99.8% of full length.
- **Bend direction is not settled in the model:** the middle joint lies towards the tail in 29 to 62% of frames,
  depending on the leg, so the cycle sets it: elbows towards the tail, knees towards the head.
- **The paws sit about 0.10 m behind** their shoulder or hip in the cat's frame (the body is pitched about 12 degrees
  nose up), and the cycle keeps that stance.
- **Speed** (shoulders and hips over the room): median 0.068 m/s; 38% of frames below 0.05 m/s; 90th percentile
  0.38 m/s.
- **Standing:** shoulders and hips stay above 85% of their usual height in 97% of frames (no lying down); all four
  paws are within 2 cm of the lowest point in only 19% of frames (dangling paws).

**The cycle** (`cluster/gait_cycle.py`, computed offline at 30 steps a second; the page reads the paw paths and bends
the legs itself):
- **Stance:** a paw that is down stays exactly where it landed, on the ground. The body is lowered so that each leg is
  at about 95% (front) and 92% (hind) of its full length at mid-stance, giving it room to sweep.
- **Swing:** a paw lifts when the body has carried its shoulder or hip about 5 cm past it, or when it can no longer
  reach. It swings for 0.3 s (cats' swing time, about constant across speeds), 2.5 to 3.5 cm up, to land ahead by
  about the distance the body covers in the next 0.3 s.
- **Order:** at most two paws in the air, never both front or both hind at once; the paw furthest behind goes first.
- **Turning in place and seam snaps** move the shoulders and hips, so the paws step too.
- **Playback:** on the copy's walks only, at 30 steps a second, with the cat's place and facing interpolated between
  the 10 Hz samples (no interpolation across a turn of over 45 degrees in 0.1 s). The recordings and the grey cat are
  unchanged. `gait=0` in the address shows version 22's walks.

**Pass rules (all needed before the before/after is shown):**
- **G1, the ruler:** on the 16 free walks, drawn side-on as before, the median leg motion is at least 0.45 (real cats
  0.60) and the median slip at most 0.25 (real cats 0.16, version 22 0.44).
- **G2, paws in 3D:** on all copy walks at the 10 Hz samples, paws that are down and slide 1.5 cm or more in 0.1 s
  are at most 3% of down steps (version 22: 9%). Same rule as before: down means within 2 cm of the cat's lowest
  point.
- **G3, the skin:** the share of mesh edges stretched by over 30% is at most 1 point above version 22's walks; and a
  sheet of each leg's most bent frames shows no tears by eye.
- **G4, the user:** one before/after of the same walk (version 22 and the cycle) in the browser pane. The user's
  verdict decides.

**Stop rule:** at most two rounds of adjustment if G1 to G3 fail (crouch depth, step size, smoothing the front legs'
skin weights). If they still fail, stop and report; version 22 stays. If G4 is not "clearly better", version 22 is
final and the walking work ends.

**Compute:** CPU only. Cluster jobs that need a GPU allocation for scheduling come out of the exploration hours (about
2 left).

**Building it (2026-11-18), before the first check.** Two mistakes in the builder were found from its own read-outs
and fixed; they were not adjustment rounds.
- **g1:** the "standing" paw place was the model's median paw, which dangles about 12 cm behind and above where a
  straight leg reaches; the body hit its 5 cm lowering limit and a quarter of stances could not reach the ground.
- **g2:** paws were planted on the room's floor whenever the cat was within 8 cm of it, but version 22's cat often
  floats a little above the floor. Its lowest point is always a paw on the walks (`gait_lowest.py`), so the ground is
  now that lowest point, smoothed over 0.5 s and never below the floor. **g3:** body lowered 1.3 cm (median), 14% of
  stance steps out of reach (they lengthen up to 5%, then step).

**First check (g3; job 12437959 and a login-node run, identical G2): G1 and G2 pass, G3 fails.**
- **G1, the ruler: pass.** On version 22's 94 walk windows and cameras: leg motion 0.62 (real clips 0.60, version 22's
  walks 0.34); slip 0.10 (real 0.16, version 22 0.36). For information, P(real) 0.74 (real 0.86, version 22 0.14).
- **G2, paws in 3D: pass.** Down paws sliding 1.5 cm or more in 0.1 s: 1.4% of 128,035 down steps (version 22,
  measured the same way: 9.2% of 81,382).
- **G3, the skin: FAIL.** Edges stretched over 30%: 17.6% against version 22's 14.1% on the same 200 frames (+3.6
  points; limit +1). The sheet of each leg's three most bent frames shows flaps of flank skin torn away at the
  shoulders and hips. Middle-joint bend while down: median 33 to 60 degrees by leg (version 22: 5 to 11), up to 140
  to 163 degrees at the worst frames.
- **Cause read from the numbers:** paws were planted further forward than the model's legs point (SLANT 0.3), so the
  whole upper leg swung forward and dragged the flank; and nothing limited the bend when a target came close.

**Round 1 of 2 (g4):** paws stand where the model's own legs point (SLANT 1.0), and the elbow or knee bends at most
70 degrees (the paw then stays short of a target that is too close). The sheet now shows version 22 at the same
frames beside the cycle.

**Round 1 result (g4; job 12445228): G1 passes, G2 and G3 fail.**
- **G1:** leg motion 0.58, slip 0.12 (pass); P(real) 0.62 for information.
- **G2: FAIL.** 4.8% of down steps slide 1.5 cm or more (limit 3%): where a target comes closer than the 70-degree
  bend allows, the paw now slides.
- **G3: FAIL.** +2.6 points (16.7% against 14.1%). By region (`gait_stretch_where.py`): body skin near the hips and
  shoulders +0.97, right hind leg +0.65, left hind +0.51, left front +0.36, right front +0.09. Version 22 itself has
  10 to 17% of edges stretched over 30% in every region. On the sheet, version 22 shows the same flank flaps at those
  frames; the cycle's tears are a little worse, mostly at the hind legs.
- **Why the hind legs:** their target was 92% of full length (front 95%), and the body was lowered by the mean of
  what the four legs needed, which over-bends the leg that needed least (right hind: median extension 0.87 against
  its 0.92 target; bend median 60 degrees).

**Round 2 of 2 (g5):** every leg at 95% at mid-stance; the body still lowered by the mean. A first round-2 variant
lowered the body only as far as the leg that needed least; its builder read-outs showed 36% of stance steps out of
reach (round 1: 15%) and the body raised 2 cm in a tenth of frames, so it was discarded before its checks ran (job
12445239 cancelled after 17 s; kept as `experiments_gait/g5_min_discarded`).

**Round 2 result (g5; job 12445240): G1 passes, G2 and G3 still fail. By the stop rule, the adjustment rounds end
here and version 22 stays.**
- **G1:** leg motion 0.57, slip 0.13 (pass); P(real) 0.64 for information (version 22's walks 0.14, real 0.86).
- **G2: FAIL.** 4.4% of down steps slide 1.5 cm or more (limit 3%; version 22: 9.2%).
- **G3: FAIL.** +2.4 points of edges stretched over 30% (limit +1). By region: body skin near hips and shoulders
  +0.86, right hind +0.61, left hind +0.49, left front +0.33, right front +0.07.
- **Leg bend while down,** median by leg: 34, 29, 38 and 57 degrees (version 22: 5 to 11). The right hind still
  bends most (extension 0.88): one lowering for the whole body cannot suit legs whose shoulders and hips sit at
  different heights (the body is pitched about 12 degrees nose up).
- **Root cause, as far as the checks show:** bending the legs as much as a walking cat does exposes the rig's skin
  weights around the hips and shoulders (as the theory report predicted); and the body's height and pitch come from
  the behaviour model, so the legs absorb all the differences. Fixing either means changing the rig's weights or the
  body's pitch, both outside this try.
- **Not shown to the user yet (G4):** the gates were set before the before/after. The page code for the cycle is in
  `viewer/index.html` (it plays the cycle only if `data/gait.*` is present; without it the page is version 22), and
  the cycle's data are in `experiments_gait/g5` on the cluster.

**G4, the user's verdict (2026-11-19): "Clearly better".** Shown once as a side-by-side video of Walk 9, 46 to
56 s (version 22 left, the cycle right, room hidden, grid floor; full and half speed;
`reports/walking_cycle_2026-11-19/`). The live browser view was tried first, but its following camera kept ending up
behind the scan's walls. The user chose the cycle although G2 and G3 missed their gates (as with version 18). Next:
version 23 prepared on the page with the round-2 cycle (g5), and published only on the user's go-ahead.
- **Version 23 prepared (2026-11-19), not yet published:** `viewer/index.html` plays the round-2 cycle (g5) on the
  copy's walks; `viewer/data/gait.json`, `gait.b64.0.txt` (5.9 MB) and `gait_test.json` added (data folder not in
  git). The "At home" note's planted-paws paragraph is replaced by one on the walking cycle, with its misses stated.
  Checked in the preview: the page's leg bending matches the cluster's within 0.0007 (bone matrix entries) on 15
  steps; no console errors; Walk 9 plays; the grey and white cat and the recordings play as before; `gait=0` gives
  version 22's walks.
- **Published as version 23 (2026-11-19)** on the user's "Publish it": page and `data/gait.json`, `data/gait.b64.0.txt`
  added to https://claude.ai/artifact/44Q8eUm9Pn4bnqZ6Xsvyrx (artifact version id 1795146214-fbf5; 61 files).
  `gait_test.json` stays local, as `feet_ik_test.json` does.

## Smoothing the skin weights (2026-11-19; checks and stop rule written before building)

After version 23 the user asked which of the theory report's mathematics had been used, then said "do skin-weight
smoothing". The walking cycle missed G3 (+2.4 points of edges stretched over 30%, mostly at the hind legs and the
skin round the hips and shoulders). The rigging notes put tearing under bending down to abrupt changes in the skin
weights between neighbouring vertices, which smooth weights remove.

**Method** (`cluster/skin_smooth.py`, CPU, no new installs): heat diffusion of the 25 bone weights over the body's
surface, implicit step (M + t C) W' = M W with the cotangent Laplacian C and lumped areas M, on the legs and the skin
around them (vertices with any weight on a leg bone, widened by RING rings of neighbours). The neck vertices the head
graft reshapes (2,215, `a2s_graft` body_idx) and their neighbours are held fixed, so the head seam cannot change; so
is everything outside the region. Then each vertex keeps its four largest weights (as the page needs), renormalised.
Round 1: t = 1e-4 m^2 (about 1.4 cm of spread), RING = 4. Only the body's weights change (`a2s_cat`); the page code,
shape, coat and the grafted head do not. The walking cycle is rebuilt with the new weights (its paw points come from
them).

**Pass rules (all needed before renders are shown):**
- **S1, the skin on the walks:** edges stretched over 30% on the same 200 walk frames, new weights with the rebuilt
  cycle, at most 1 point above version 22 (the original G3 gate; version 23 is +2.4).
- **S2, paws:** down paws sliding 1.5 cm or more no worse than version 23 (4.4%) by more than 0.5 point.
- **S3, the recordings stay as they are:** on 300 recording frames, the share of edges stretched over 30% no more
  than 0.2 point above the old weights, and vertices moved by the new weights at most 1 mm (median) and 1 cm (99th
  percentile).
- **S4, the ruler:** G1 still passes (leg motion 0.45 or more, slip 0.25 or less).
- **S5, by eye:** each leg's most bent frames, old and new weights side by side: no new tears.
Then before/after renders for the user, and publishing only on the user's go-ahead.

**Stop rule:** at most two rounds (diffusion time and ring width). If S1 or S3 still fails, stop; version 23 stays.

**Round 1 (s1; job 12447236): all checks pass, S1 narrowly.** Smoothed 12,098 of 15,695 vertices (the leg bones'
weights reach over most of the body); 2,455 neck vertices held. Weight change median 0.02 (90th percentile 0.07);
the largest weight jump across an edge in the region fell from 0.45 to 0.33 (99th percentile 0.20 to 0.17).
- **S1: pass.** Edges stretched over 30% on the 200 walk frames: version 22 14.08%, version 23 16.44% (+2.36), new
  weights with the cycle rebuilt on them 15.02% (+0.94; limit +1).
- **S2: pass.** Down paws sliding 1.5 cm or more: 4.46% (version 23: 4.41%).
- **S3: pass.** Recordings: edges stretched over 30% 14.92% with the old weights, 13.30% with the new (1.6 points
  better); vertices moved 0.43 mm (median), 8.0 mm (99th percentile), 61 mm at the most (one vertex, one frame).
- **S4: pass.** Leg motion 0.57, slip 0.13; P(real) 0.62.
- **S5: pass by eye.** Each leg's three most bent frames show no new tears; side by side with version 23 the
  difference is hard to see at these frames (the jagged lines there are in both).
- **Next:** before/after renders for the user; publish only on the user's go-ahead.
- **Correction, the same evening: round 1 fails S5.** In a close, textured side view of Walk 9 (the before/after
  video), the smoothed version shows black cracks across the coat that version 23 does not. The plain grey sheet at
  thumbnail size hid them. Cause: the mesh carries duplicate vertices along the coat's texture seams; the smoothing
  treated each copy separately, gave them different weights, and the seams open when the cat moves.
- **Round 2 of 2 (s2):** the same smoothing on the welded mesh (vertices at the same place are one point, and every
  copy gets the same weights). Added check, after seeing the defect: **S6, seams stay shut** - copies of a vertex
  stay within 0.1 mm of each other in every checked frame.

**Round 2 result (s2; job 12447675): S1 fails. By the stop rule the smoothing ends here and version 23 stays.**
- **S6, seams: pass.** 693 seam copies welded (15,695 vertices, 15,002 points); in the original weights every copy
  already matched its twin exactly, which confirms round 1's cause. Largest gap between copies in 150 frames: 0 mm.
- **S1: FAIL.** Edges stretched over 30% on the 200 walk frames: 16.08% against version 22's 14.08% (+2.0; limit +1;
  version 23 +2.36). With the seams kept shut, the smoothing removes only about a sixth of the cycle's extra stretch.
  Round 1's +0.94 came with the seams torn open and does not stand.
- **S2: pass** (4.45%). **S3: pass** (recordings 0.5 point better; vertices moved 0.39 mm median, 6 mm at the 99th
  percentile). **S4: pass** (leg motion 0.57, slip 0.13).
- **What this says:** the extra stretch under the walking cycle does not come mainly from abrupt weight changes
  between neighbouring vertices. The rigging notes' other causes remain: joint pivots that are not at the cat's real
  elbows, knees and hips (the learnt skeleton was never moved there), and linear blend skinning itself at large
  bends. Fixing those means moving the joints or changing the skinning method, not smoothing.
- Nothing published; the page and its data are unchanged. Renders and the round-1 videos (with the cracks) are in
  `reports/skin_smoothing_2026-11-19/`.

## Real cats' gait from video (2026-11-20; checks and stop rule written before building)

The user asked what big generative video models can teach the project; the main lesson offered was that their cats
walk well because they learned how cats move from huge amounts of real video. The cheapest way to borrow that without
generative AI: measure real cats' gait from video with a 2D pose model, and tune the walking cycle to it. The user
chose "Real-video gait stats".

**Data:** Lab4D's public cat-85 archive, already on the cluster: every clip's raw frames at 30 a second
(`JPEGImagesRaw`; e.g. 0071 has 99 raw frames against 14 kept). Any cat counts (gait is a species trait); the
rolling clips 0078 to 0080 are left out. Walking stretches come from the ruler's side-view screening.

**Pose model:** SuperAnimal-Quadruped (Ye et al., Mathis lab, EPFL; HRNet-w32 trained on Quadruped-80K, side-view
quadrupeds), `pose_model.pth` from huggingface.co/mwmathis/DeepLabCutModelZoo-SuperAnimal-Quadruped (about 160 MB).
Licence: modified MIT, academic and non-commercial use only. Run in the existing environment (torch 2.4, timm 1.0.9),
no new installs; the cat is cropped with the set's own outlines (interpolated between kept frames), so the detector
is not needed. Download only after the user approves it; it goes to the cluster, not the Mac.

**Measures** per walking stretch, in body lengths so they do not depend on the camera: speed; each paw's fore-aft
position relative to its shoulder or hip; stride period; stance (paw moving back relative to the body) and swing
times; duty factor; stance sweep; lift height; phase offsets between legs.

**E1, trust check before any use:** keypoints sit on the paws on a sheet of 20 random frames (by eye), and the
measured stride frequency and duty factor fall in the published range for walking cats (stride 0.6 to 2 per second,
duty factor 0.5 to 0.8). If not, the measures are not used and the work stops.

**Then:** the cycle's settings (swing time, stance sweep, lift, landing lead, leg order) are set from the measures as
functions of speed, and the cycle is rebuilt. **Pass rules** against version 23 on the same windows: the ruler's
P(real) at least 0.05 higher than 0.62, leg motion still at least 0.45; G2 (paws sliding 1.5 cm or more) at most
4.9%; G3 no worse than version 23's +2.4 points by more than 0.3. Then a before/after video; the user decides.
**Stop rule:** one round and one adjustment. If P(real) does not rise by 0.05, stop; version 23 stays.

**Pose model downloaded** (user approved, 2026-11-20): `pose_model.pth`, 159,622,507 bytes, sha256 6c9c66d4...e8e5df,
to `models/superanimal_quadruped/` on the cluster with its README (licence). Loaded with weights only (the file's one
non-tensor type, a YAML float, is stood in by a plain float; `cluster/sa_load.py`). Run with timm's hrnet_w32
(features only, stride-4 branch) and the checkpoint's 1x1 heatmap head, as DeepLabCut's own config and code describe
(`cluster/sa_pose.py`).

**Keypoints (job 12465631):** 17 clips with side-view walking (0055 to 0084, rolling clips out), 2,844 raw frames at
30 a second. Paw scores median 0.76 to 0.93 on 15 clips, 0.42 on 0081 and 0084. **E1 by eye: pass for standing and
walking side views** (thigh, knee and paw points sit on the legs on 0055, 0060, 0066, 0068, 0069); wrong on sitting,
lying and steep top-down frames (0056, 0059, 0067, 0070), which the walking filter leaves out.

**Measures (`cluster/gait_stats.py`): 77 strides from 10 clips. E1 numbers: pass** (stride frequency 0.75 a second,
duty factor 0.75).
- Speed 0.19 to 0.64 body lengths a second (thirds: 0.19, 0.32, 0.64). These are cats ambling at home.
- Swing time 0.30 s median (0.28, 0.33, 0.20 s by speed third); stance 1.03 s (1.23, 1.00, 0.30); duty 0.75 (0.81,
  0.75, 0.66).
- Stance sweep 0.29 body lengths (0.25, 0.34, 0.26); paws land 0.18 body lengths ahead of the thigh point and lift
  0.085 behind it (about 4.4 cm either side of mid-stance on our cat).
- Paw lift 0.031 body lengths (about 1 cm on our cat).
- Each front leg's swing starts 0.16 of a stride after the same side's hind leg's (27 pairs): a lateral-sequence walk.
- Our cat's body (neck base to tail base) is about 0.33 m. The cycle already matches stride length (0.45 against
  0.42 body lengths), stance sweep and step rate; it does not match swing time at speed, leg order, or lift (2.5 to
  3.5 cm against about 1 cm).

**The retune (g6), set from these measures:** swing time 0.30 s up to 0.35 body lengths a second, shortening to 0.20 s
at 0.65 and above; lateral sequence (a front paw lifts only after the same side's hind paw has lifted since its own
last step, unless it is out of reach or 1.6 times past its lift point); lift a constant 2.5 cm.
- **Lift is a compromise, decided before the checks:** real cats lift about 1 cm, but G2 counts any paw within 2 cm of
  the lowest point as down, so a realistic swing would be scored as sliding. Lift stays at the lowest value G2
  leaves alone (2.5 cm), and the rule is not changed after seeing data.

**Retune result (g6; job 12465754): the ruler did not move. Adjustment (g7; job 12465916): worse. By the stop rule
the retune ends; version 23 stays.**

| | P(real) | leg | slip | G2 (limit 4.9%) | G3 (limit +2.66) |
|---|---|---|---|---|---|
| version 23 (g5) | 0.637 | 0.575 | 0.129 | 4.4% | +2.36 |
| g6: swing time by speed, leg order, 2.5 cm lift | 0.624 | 0.582 | 0.113 | 5.4% FAIL | +2.15 |
| g7: leg order and 2.5 cm lift, 0.3 s swings | 0.578 | 0.563 | 0.125 | 6.6% FAIL | +2.31 |

(Version 23's P(real) is 0.637 to three places; "0.62" earlier was rounded down. The gate was 0.67.)
- **Why G2 got worse:** the leg-order rule makes a front paw wait for its hind paw, so it is dragged further before
  it steps.
- **The doubtful measure:** shorter swings at speed (0.20 s) contradict treadmill studies (about 0.3 s at all walking
  speeds); with few fast strides here it may come from how swing and stance were split. g7 dropped it; no gain.
- **What the measures did show:** the cycle's stride length (0.45 against 0.42 body lengths), stance sweep and step
  rate already match real cats ambling at home. The remaining difference is elsewhere.
- **Where the ruler still sees a difference** (window medians, real / version 23): torso motion 0.53 / 0.34 (the
  body half as lively; real handheld footage may inflate this); hind-leg swing 0.27 / 0.46 (our hind legs swing
  far more than real ones); mean front-leg drift -0.27 / -0.09; front-hind correlation -0.21 / -0.47. Slip (0.16 /
  0.13) and overall leg motion (0.60 / 0.58) now match.

**"Try hind steps" (user, 2026-11-20): stopped before building, on a check of the premise.**
- The ruler's hind-swing measure is the spread of the hind leg region's motion against the body, over the body's
  travel. Real cats: front 0.26, hind 0.27; version 23: front 0.26, hind 0.46.
- In the real-cat video strides, hind legs step exactly like front legs: swing 0.30 s both, sweep 0.27 against 0.29
  body lengths, lift 0.034 against 0.031, duty 0.73 against 0.77 (41 hind and 36 front strides). Every leg must also
  cover the body's travel each stride, so hind steps cannot be smaller without stepping more often than the front.
- So smaller hind steps would move the gait away from real cats to suit one ruler measure, the risk noted when the
  ruler was built. Since our front legs already match on this measure with the same step settings, the extra hind
  motion more likely comes from how the hind region moves: the hind legs bend most (right hind median 57 degrees,
  extension 0.88, because one lowering serves the whole body pitched 12 degrees nose up), and the skin round the hips
  stretches most (G3). Put to the user before building anything.

## Levelling the body on the walking cycle (2026-11-20; checks and stop rule written before building)

The user chose "Level the body" over smaller hind steps. On version 23 one lowering serves all four legs while the
body is pitched about 12 degrees nose up, so the hind legs end up over-bent (right hind median 57 degrees, extension
0.88; front legs about 30 degrees) and the hip skin stretches most.

**Method** (`cluster/gait_cycle.py`, LEVEL=1): per frame, how far the front pair and the hind pair each need to come
down for 95% extension; the difference becomes a pitch of the whole body about the middle of its shoulders and hips
(at most 10 degrees, smoothed over 0.5 s), the mean becomes the lowering. Every bone turns rigidly with it, so head,
spine and tail tilt together and the grafted head and coat are untouched. The page applies the same pitch per 1/30 s
step before bending the legs (new `pitch` and `pivot` in `gait.*`).

**Pass rules** (against version 23 on the same frames and windows):
- **L1, the point:** the right hind's median bend while down at most 45 degrees, and the four legs' median bends
  within 15 degrees of each other.
- **L2, the ruler:** P(real) at least version 23's 0.637; leg motion at least 0.45 and slip at most 0.25.
- **L3, paws:** G2 at most 4.9%.
- **L4, skin:** G3 no worse than version 23 (+2.36 points).
- **L5, by eye:** a close, textured side video of Walk 9 shows no new cracks or tears, and the cat looks level.
Then a before/after for the user; publishing only on the user's go-ahead.
**Stop rule:** one round and one adjustment; if L1 to L5 do not all pass, stop and version 23 stays.

**Results: round 1 (g8, pitch only; job 12466533) and the adjustment (g9, pitch and roll fitted together by least
squares to the four legs' needs; job 12466582). Both miss; by the stop rule version 23 stays.**

| | right hind bend | bends (front -x, front +x, hind -x, hind +x) | P(real) | G2 | G3 |
|---|---|---|---|---|---|
| limit | <= 45 deg, spread <= 15 | | >= 0.637 | <= 4.9% | <= +2.36 |
| version 23 (g5) | 57.3 | 34, 29, 38, 57 | 0.637 | 4.41% | +2.36 |
| g8: pitch (median 2.3 deg nose down) | 52.9 FAIL | 39, 37, 35, 53 (spread 18 FAIL) | 0.595 FAIL | 3.96% | +2.25 |
| g9: pitch 1.3 deg, roll -5.5 deg median (to the 10 deg cap a fifth of the time) | 46.2 FAIL | 40, 33, 39, 46 (spread 14) | 0.644 | 3.43% | +2.45 FAIL |

- The imbalance was mostly sideways (right hind against left hind), not front to back; fitting the roll as well
  evened the legs most (spread 28 to 14 degrees) and cut paw sliding to 3.4%, but missed L1 by 1.2 degrees and L4
  by 0.09 points. L5 (the close textured video) was not run, since L1 and L4 had already failed.
- The page code now understands `pitch`, `pivot` and `roll` in the cycle's data; the published data has none, so the
  live page is unchanged (version 23).
- **Shown anyway on the user's "Show me once" (2026-11-20):** close, close at half speed and wide side-by-side videos
  of Walk 9 (`reports/body_levelling_2026-11-20/`); checked by eye first (L5): no cracks or tears. **User's verdict:
  "Not clearly better". Version 23 stays; the levelling is not published.** The page file keeps the (unused) code for
  `pitch`, `pivot` and `roll`; it differs from the published version 23 only by that.

## Contact shadows at home (2026-11-21; checks and stop rule written before building)

After the levelling was not clearly better, the user asked for the next takeaway from generative models. The answer
given: they learn the whole picture at once (every part agreeing with every other and with the room), while our cat
is assembled from separate pieces, and at home it casts no shadow (the Body tab has a soft one; the home view has
none). Three options were offered: contact shadows, body motion tied to the steps, a video model as the judge. The
user chose "try 1 first": contact shadows.

**Method (page only, no new data):** at home, a soft shadow under the body (oval, along the cat's heading, sized from
its posed outline) and a small dark spot under each paw. The ground is the room floor when the cat's lowest point is
within 8 cm of it, else the lowest point itself (on furniture). A paw's spot is darkest when the paw is down and
fades out by 3 cm above the ground; the body shadow fades as the whole cat rises (a jump). Paw spots for Pikachu
only (its paw bones are known); the grey cat gets the body shadow. Recordings and walks alike.

**Checks:**
- **C1, by eye (before showing):** stills and a short sequence of a walk, a recording and a moment on furniture:
  shadows sit under the cat and its paws, follow them without flicker, and do not float or show through furniture.
- **C2, cost:** no console errors; the shadow update adds under 2 ms per pose on this Mac.
- **C3, the user:** one before/after (with and without shadows, the same walk and a recording).
**Stop rule:** one round and one adjustment (size and darkness); if the user does not find it clearly better, the
shadows are dropped and version 23 stays.

**Result (2026-11-21): the shadows are drawn in the right place but can hardly be seen; C1 fails on visibility. By
the stop rule they are not shown as a before/after; version 23 stays.**
- **Round 1:** shadows on the page's single floor height. Not visible: that height is 7 cm below the scanned floor
  around the cat, so the cat counted as "on furniture" and the shadow sat at its paws.
- **Adjustment (size and darkness, plus the floor):** a 4 cm height map of the scanned floor built at load, and
  darker shadows (body 50%, paws 70% at the centre). Two placement bugs then fixed, not counted as rounds: each
  cell's floor taken as the lowest scan point (stray points lie below the floor, so shadows sank out of sight); now
  the median of the cell's lowest 3 cm, the oval at the highest floor under its footprint, never below the cat.
- **Painted solid red, the oval and the paw spots sit exactly under the cat and its paws.** With the real shading
  they are nearly invisible: seen from above (the page's usual view) the cat covers the darkest part, and the faint
  rim that shows is lost on the patterned wooden floor. The scan's floor also undulates by a few cm, so a flat oval
  partly dips under it.
- **C2: pass** (no console errors; no measurable added time per pose, about 2 ms with or without).
- **What would make them visible:** a light direction (a cast shadow falling to one side, as the room's window light
  would throw it), or shadows painted onto the scan surface itself. Both are a different design from soft contact
  shadows, so they are left for the user to decide.
- The page file carries the shadow code (unpublished; `shadow=0` turns it off). `work/pets/preview_tools/` now holds
  the preview helpers (the scratch folder is cleared now and then).

## A cast shadow at home (2026-11-21; checks and stop rule written before building)

The user chose "Try cast shadow": a shadow thrown to one side by the room's light, which a viewer can see (the soft
contact shadow straight below was hidden by the cat).

**Light direction (an assumption, from the scan):** the brightest high parts of the room scan (brightness in the top
2%, over 0.6 m up) cluster on the two end walls, most at z = -4.4 m, 1.1 to 1.8 m up (likely windows). One fixed
light from that end, 45 degrees up, so shadows fall towards +z. Whether this matches the light baked into the scan's
colours is judged by eye.

**Method (page only):** three.js shadow mapping. A directional light of zero brightness (it changes nothing on the
cat's shading, only casts), soft edges (PCF soft), a 1024 map over 1.4 m around the cat, following it. The cat
(body, grafted head; the grey cat too) casts; a shadow-only catcher plane (2.4 m, 35% darkness) lies under it on the
scanned floor (the floor height map, at the highest floor under the cat, never below it). The paw contact spots stay;
the soft oval goes (the cast shadow replaces it). `shadow=0` turns all of it off.

**Checks:**
- **C1, by eye:** from the page's usual high view, at a walk, a recording and a moment near furniture: the shadow is
  visible beside the cat, joins it at the paws, is soft, falls away from the bright end wall, and no patch floats in
  the air or shows through furniture.
- **C2, cost:** no console errors; rendering a frame takes at most 3 ms longer with the shadow.
- **C3, the user:** a before/after video from the page's usual view.
**Stop rule:** one round and one adjustment (direction, softness, darkness); if not clearly better to the user, it
is dropped and version 23 stays.

**Built (2026-11-21).** Round 1: a 35%, PCF-soft shadow: clearly visible beside the cat in the high stills, but crisp
and dark like sunlight, while the scan's own lighting is diffuse (no hard shadows from its furniture). Adjustment:
25% darkness and a wide PCF filter (radius 8), so it reads as soft window light.
- **C1, by eye: pass.** From above, the shadow falls beside the cat, starts at its paws and is soft; it is hidden
  when it falls behind the cat or under the scanned cat bed; no floating patches or shadows through furniture seen
  in the stills or the recorded walk.
- **C2: pass.** No console errors; drawing a frame takes 2.9 ms without and 3.0 to 3.3 ms with the shadow.
- **Before/after for the user (C3):** Walk 9, 46 to 56 s, room visible, from above
  (`reports/contact_shadows_2026-11-21/cast_shadow_walk9_from_above.mp4`); a side-camera version is there too, but
  the bed blocks that camera for part of the walk.
- **C3, the user's verdict (2026-11-21): "Clearly better".** Version 24 prepared with the cast shadow; published only
  on the user's go-ahead.
- **Version 24 prepared (2026-11-21), not yet published:** `viewer/index.html` with the cast shadow and paw contact
  spots at home, and a paragraph on the shadow in the "At home" note (its direction stated as a guess). No data files
  change. Checked in the preview: no console errors; the walking cycle's leg bending still matches the cluster's
  (within 0.0007); the grey and white cat and a recording play.
- **Published as version 24 (2026-11-21)** on the user's "Publish it": the page file only, at
  https://claude.ai/artifact/44Q8eUm9Pn4bnqZ6Xsvyrx (artifact version id 1795273233-bd86).

## The body moving with the steps (2026-11-21; checks and stop rule written before building)

After version 24 the user asked "next?"; the suggestion was option 2 from the generative-model takeaway (the whole
body agreeing with the legs), and the user chose "Body with steps". On the walks the legs now step, but the spine,
head and tail keep the behaviour model's motion whatever the paws do; the ruler's largest remaining gap is the body's
own motion (torso 0.34 against real cats' 0.53).

**Measures (from the SuperAnimal tracks already on the cluster, `cluster/body_stats.py`):** over each real stride
(from one hind paw's lift to its next), in body lengths and the body's own frame (so a hand-held camera does not
matter): the head's rise and fall against the neck base; the shoulders' and hips' height above their standing paws;
the tail's angle against the body. Each averaged over strides into one waveform over the stride. Side to side sway is
not visible in side views and is left out.
- **E1, trust:** at least 30 strides with all the needed points confident; waveforms consistent between the two
  halves of the strides (split-half correlation at least 0.5) for any motion that is used. A motion that fails is
  left out.

**Method:** the cycle builder drives the measured waveforms with its own stride phase (from the hind paws' lifts),
fading them out when the cat stands: shoulder and hip rise and fall as a lowering and pitch of the whole body (the
page already applies these), the head as a nod of the head bone about its joint, the tail as a swing of the tail
bones about the tail base. The page gets per-step `head` and `tail` angles.

**Pass rules (against version 24 on the same windows and frames):**
- **B1, the ruler:** torso motion at least 0.40 (version 24 0.34, real 0.53) and P(real) no lower than version 24's
  by more than 0.01; leg motion at least 0.45, slip at most 0.25.
- **B2, paws:** G2 at most 4.9%. **B3, skin:** G3 at most +2.66 points.
- **B4, by eye:** a close, textured side video of Walk 9: the head and tail move like a walking cat's, not
  mechanically, and no new cracks.
Then a before/after video for the user. **Stop rule:** one round and one adjustment; if any of B1 to B4 fails after
that, stop and version 24 stays.

**Measures (`cluster/body_stats.py`, 2026-11-21):** 77 strides, of which the hind strides give the waveforms.
| motion | strides | split-half r | mean waveform, half peak-to-peak | single strides (median) | E1 |
|---|---|---|---|---|---|
| head (nose against neck base) | 39 | 0.84 | 0.017 body lengths (about 0.55 cm on our cat) | 0.036 | used |
| shoulders (height over the standing paw) | 33 | 0.96 | 0.014 (about 0.47 cm) | 0.033 | used |
| hips | 34 | 0.18 | 0.017 | 0.046 | not used |
| tail angle | 34 | 0.47 | 4.3 degrees | 12.1 degrees | not used |
- The head and shoulder waveforms rise and fall twice a stride (once per footfall); the averaged waveform is about
  half the size of a single stride's motion because the strides' timing varies.
- (A slip fixed: `body_stats.py` first imported `gait_stats.py`, which re-ran its measurement on import and rewrote
  `gait_stats.json`; the content was the same.)
- **Before building:** the usable motions are half a centimetre on our cat. Put to the user whether to build.

**Found before building (2026-11-21): the walking cycle has been moving the head with the left front leg.**
- In the body's skinning, the head and neck (2,215 vertices, and 2,670 of the grafted head's 2,776) are moved mainly
  by bone 9. The leg chains (from `feet_ik.py`) treat bone 9 as the middle bone of the -x front leg, so every bend of
  that leg turns it: by 14 degrees (median), 32 (90th percentile) on the walking cycle. No vertex is mainly moved by
  bone 3 (the joint tree's head bone); the learnt skinning does not follow the joint tree there.
- Measured on 300 walk frames, the head region moves against the spine by 30 mm (median), 56 (90th percentile), 71
  (99th) on the walking cycle (versions 23 and 24), against 4, 23, 36 mm with version 22's planted paws.
- **Fix, part of this try:** the leg bending leaves bone 9 alone (`protect` in the cycle's data; the page and the
  builder skip it), so the head keeps the behaviour model's own head motion. The -x front leg then bends through
  bones 8, 10 and 11 only.
- **The head nod** therefore turns bone 9 (not bone 3), about the head joint (joint 4) as bone 9 carries it.
- The before/after will show both changes together (head steady against the leg, then nodding with the steps); B1
  to B4 stand as written.

**Round 1 (g10; job 12474694): B1 fails.** Head kept off the leg bending, head nod and shoulder bob at the averaged
waveforms' size (nod about +-3 to 4 degrees; body motion on 40% of walk steps, where the cat is striding).
- **B1: FAIL.** Torso motion 0.293 (needs 0.40; version 24 0.337, real 0.534). P(real) 0.629 (version 24 0.637,
  within the allowed 0.01); leg 0.564, slip 0.130.
- **B2: pass** (4.52%). **B3: pass, better than version 24:** +1.95 points (version 24 +2.36): the neck is no longer
  dragged by the leg.
- **Why torso fell:** version 24's head swung 3 cm (median) with the left front leg, and the ruler's torso measure
  counted that as liveliness. Keeping the head off the leg removes more torso motion than half-centimetre bobs add.
- **Adjustment (g11):** each waveform scaled to a single stride's size (head x2.15, shoulders x2.34): averaging strides
  whose timing varies had halved them. Nod about +-7 to 9 degrees.

**Adjustment result (g11; job 12474715): B1 still fails; by the stop rule the body motion is not used, and version
24 stays.**
| | torso | P(real) | leg | slip | G2 | G3 |
|---|---|---|---|---|---|---|
| version 24 (g5) | 0.337 | 0.637 | 0.575 | 0.129 | 4.41% | +2.36 |
| g10: head kept off the leg, averaged bob | 0.293 | 0.629 | 0.564 | 0.130 | 4.52% | +1.95 |
| g11: head kept off the leg, single-stride bob | 0.314 FAIL (needs 0.40) | 0.628 | 0.575 | 0.143 | 4.89% | +1.96 |
- B4 (the close video) was not run, since B1 had already failed.
- **What stands from this try:** the published walks (versions 23 and 24) swing the head about 3 cm with the left
  front leg's steps (bone 9 moves both). Keeping bone 9 off the leg bending steadies the head and lowers the skin
  stretch (+1.95 against +2.36), at no cost the ruler can see (P(real) 0.629 against 0.637, within noise). It is a
  fix of its own, not part of the body motion; put to the user.

## The head kept off the left front leg, on its own (2026-11-21; the fix found in "The body moving with the steps")
The user chose "Show head fix". Built as g12: version 24's walking cycle (g5's settings) with bone 9 left out of the
leg bending (PROTECT = (9,)), no head nod or body bob. Same steps and paw places as version 24 (steps per paw
identical). Job 12475835 for the checks (CPU on a GPU node, a few minutes).
| | head moved against the spine by the leg bending, mm q50/q90/q99 | bone 9 turned, degrees q50/q90/q99 | P(real) | leg | slip | G2 | G3 |
|---|---|---|---|---|---|---|---|
| version 22 (planted paws) | 4.1 / 23.1 / 35.5 | | | | | | |
| version 24 (g5) | 30.4 / 55.8 / 70.7 | 13.6 / 31.6 / 44.3 | 0.637 | 0.575 | 0.129 | 4.41% | +2.36 |
| head fix (g12) | 10.4 / 16.0 / 20.7 | 0.0 / 1.4 / 1.8 | 0.627 | 0.559 | 0.124 | 4.41% | +1.95 |
(the head measure on the same 300 random walk frames, seed 2, as when the flaw was found; the head still moves about
1 cm with the other leg bones that share its skin.)
- The page reproduces g12's bones (local self-test with gait_test.json from g12: worst 7e-4; the same test against
  version 24's data gives 0.08 to 0.67, so it does tell them apart).
- Before/after video for the user: `reports/head_fix_2026-11-21/head_fix_walk9.mp4` and `_half_speed.mp4` (Walk 9,
  frames 460 to 560, side view 1.3 m back, grid floor, cropped round the cat). Version 24 bows the head towards the
  chest as the left front leg bends; the fix holds it up.
- **User verdict (2026-11-22): "Clearly better".** Version 25 prepared, not yet published: viewer/index.html is version
  24 plus `protect` in the leg bending (the unused head-nod code from the body-motion try taken out again), and the
  At home note says the head bone is left out of the bending (head swing about 3 cm before, about 1 cm now), with
  the leg measure 0.56 and the skin stretch 2.0 points. viewer/data/gait.* now hold g12 (version 24's g5 data stays
  on the cluster in experiments_gait/g5). Local self-test of the page against g12's bones: worst 7e-4; no console
  errors.
- **Published as version 25 (2026-11-22, user "Publish it"; artifact version id 1795342522-abb1):** index.html plus
  data/gait.json (now with "protect": [9]) and data/gait.b64.0.txt. The paw-target data is byte-for-byte version 24's
  (same sha256): the paws did not change, and the fix is the `protect` setting read by the page's leg bending.

## A video-model judge (2026-11-22; plan, checks and stop rule written before any download or build)
User: "Plan the judge" (the third generative-model lesson). Purpose: a judge that says whether a change to the cat's
walk would look more like a real cat, agreeing with the user's eye better than the hand-made ruler did (the ruler
called the head fix slightly worse, P(real) 0.627 against 0.637, and the levelling slightly better, 0.644 against
0.637; the user found the first clearly better and the second not). It screens tries; it never publishes anything.

**Design** (from research_notes/Cat animation bottleneck theory/realism_evaluation_perception.md: JEDi = V-JEPA
features + polynomial-kernel MMD, more human-aligned and sample-efficient than FVD, arXiv 2410.05203; appearance must
be equalised or render-against-photo differences drive any distance):
- **Same look on both sides:** every clip is a silhouette (white cat on black), cropped square round the cat and
  resized to 256 px. So shadows, coat and room are outside this judge; it judges motion and outline only.
- **Real cats:** the cat-85 walking clips (the 10 clips with the 77 strides found by gait_stats.py; more clips'
  walking parts if needed), raw frames at 30 fps from data/cat85/cat-85.zip. Their Lab4D outlines exist only at
  irregular kept frames (16 dense kept frames span only 0.7 s), so the outline of every raw frame comes from SAM 2.1
  (video segmentation, prompted with the Lab4D outline at the kept frames).
- **Our cat:** the same side-view walk windows as the ruler (experiments_judge/ours/walk_shown_windows.json), posed
  at 30 Hz exactly as the page poses them (gait_cycle.py's interpolation and leg bending), drawn as silhouettes with
  judge_ours.py's drawing (orthographic side cameras, the same per window for every version).
- **Clips:** 16 frames at 7.5 fps (2.1 s, about one and a half strides), windows overlapping by half.
- **Features:** V-JEPA 2 ViT-L (facebook/vjepa2-vitl-fpc64-256), encoder tokens averaged to one 1024-number vector per
  clip.
- **Score:** unbiased MMD with the polynomial kernel (as KID/JEDi) between a version's clips and the real clips; lower
  is more like real cats. A against B: the difference of the two scores, on the same windows, with a paired bootstrap
  over windows (2,000 draws, 95% interval).

**Checks (all set before building):**
- **J0, enough real clips:** at least 60 real 16-frame windows after outlines; else stop before any scoring.
- **J1, agrees with the user's past verdicts** (the three walk changes the user judged by before/after):
  V1 the walking cycle (g5) better than version 22's walks: "Clearly better";
  V2 the head fix (g12) better than version 24 (g5): "Clearly better";
  V3 the levelling (g9) against version 23 (g5): "Not clearly better".
  Pass: V1 and V2 in the right direction with the 95% interval excluding zero, and V3 not significantly better, or
  its gain smaller than both V1's and V2's. (The ruler: 1 of 3.)
- **J2, sees known defects** (an automatic ladder on g12, no rating): paws sliding forward 4, 8, 16 mm per 0.1 s while
  down (judge_ours SLIDE_MM); legs stiffened half and fully back to version 22's; poses held for 0.2 s and 0.4 s; bone
  jitter of 1, 3, 6 degrees; and the walk played backwards. Pass: in at least 4 of the 5 kinds, the score worsens at
  each step up and the strongest step is worse with the interval excluding zero.
- **J3, real is real:** held-out real clips (half the clips, split by clip) are closer to the other half than every
  version of our cat is, interval excluding zero.
- **J4, cheap enough:** scoring one new candidate takes at most 15 GPU-minutes.

**Stop rule:** build once; one adjustment allowed (clip length 32 frames, or 15 fps, or the feature averaging) if J1,
J2 or J3 fails; if any still fails, the judge is not adopted and the ruler plus the user's before/after stay. No
further tuning: three verdicts are far too few to tune against, so passing J1 is weak evidence, which is why J2 and
J3 must pass too.

**If adopted:** a try the judge does not score as better (interval excluding zero) stops before a before/after is
made, and the result is reported. Anything to be published still needs the user's verdict. Each new verdict is
logged against the judge; it is dropped if it disagrees with the user twice in a row.

**Needs from the user:** two downloads (V-JEPA 2 ViT-L model.safetensors, 1.30 GB, MIT, huggingface.co/facebook/
vjepa2-vitl-fpc64-256; SAM 2.1 small, 184 MB, Apache 2.0, dl.fbaipublicfiles.com/segment_anything_2/092824/
sam2.1_hiera_small.pt); a new Python environment on the cluster, built by the user (SAM 2 needs torch 2.5.1 or newer
and V-JEPA 2 needs transformers 4.53 or newer; the cluster's environments have torch 2.4.1 and no transformers); and
2 GPU-hours (about 0.9 of the 10 exploration hours are left). Expected use: under 1 GPU-hour.

**Approved and downloaded** (user: "Approve all", 2026-11-22): V-JEPA 2 ViT-L at revision b3c1679b (config.json,
video_preprocessor_config.json, model.safetensors 1,303,947,864 bytes, sha256 25466aef...b786b) to
`models/vjepa2-vitl-fpc64-256/`; SAM 2.1 small (184,416,285 bytes, sha256 6d1aa6f3...4d38) to `models/sam2/`
(`cluster/dl_judge.sh`). Environment script for the user: `cluster/setup_judge.sh` (env-judge: Python 3.11, torch 2.5.1
cu121, transformers 4.57.6, SAM 2 at commit 2b90b9f5 without its optional CUDA extension). The login node fe00's
scratch disk hung that morning; fe02 was used.

**Build details, fixed before any clip was scored:**
- **Our clips** (`cluster/jepa_ours.py`): the 220 ruler windows; each clip 16 frames, one every 4 steps of the page's
  30 a second (2.0 s), from the window's first frame (earlier if the walk ends first); a still orthographic camera
  (the window's azimuth and elevation, its direction from the clip's travel); one still square per clip round
  everything the cat covers (1.15 times the larger side), triangles filled at 512 px, reduced to 256 by area.
- **Versions:** the walking cycles posed at every 30 Hz step as the page does (`gait_cycle.py` with DENSE=1:
  blended bones, then the legs bent to the paw targets), rebuilt for g5 and g9 with PROTECT="" (the bone-9 guard did
  not exist then; a check compares the rebuilt gait.bin and gait_bones.npz with the published builds). Version 22's
  walks as its page showed them: no 30 Hz blending (the page only blended with the walking cycle loaded), so
  place, facing, floor shift and the planted-paw bones are held for 0.1 s.
- **Defect ladder on g12:** sliding paws through `gait_cycle.py` SLIDE_MM (each paw that is down moves forward 4, 8,
  16 mm per 0.1 s since it landed, at most 1 s; the swing starts from where it slid to; step timing unchanged); stiff
  legs: the 16 leg bones blended elementwise 50% and 100% to version 22's planted-paw bones (blended between the
  0.1 s samples, so only the legs change); held poses: everything held 6 and 12 steps (0.2 and 0.4 s); jitter: each
  bone turned about its own joint by a random axis and a normal angle of sd 1, 3, 6 degrees, new each frame, the same
  draws scaled; backwards: the clip's frames in reverse order.
- **Real clips** (`cluster/jepa_real.py`): the 17 clips of "Real cats' gait from video"; SAM 2.1 prompted with the
  Lab4D outline at a kept frame every 15 raw frames (nearest kept frame), tracked both ways, its outline compared
  with Lab4D's at the other kept frames (IoU reported); the camera's pan removed by the background's optical flow
  (Farneback, quarter size, median outside the widened outline, summed); clips start every 32 raw frames, cropped
  by the same rule as ours. **Kept if walking and seen whole:** side-on (gait_stats.py's rule) on at least half its
  raw frames and at least 2 swing starts (gait_stats.py's swing rule) inside it; an outline on all 16 frames, touching
  the picture's edge on at most 2.
- **Features** (`cluster/jepa_feat.py`): the clip and its mirror image through the V-JEPA 2 encoder (no predictor),
  tokens averaged, the two vectors averaged. **Scores** (`cluster/jepa_score.py`): as planned; J3's halves are the
  real videos in alternate name order.
- **More real clips, before any scoring:** the walking filter left 29 of the 17 clips' 63 windows (before the outline
  checks), so J0 (60) could not pass. As the design allows ("more clips' walking parts if needed"), every other clip
  of the archive with raw frames (57 more, about 8,500 frames; rolling clips 0078 to 0080 still out) gets SuperAnimal
  keypoints (`cluster/run_jepa_pose.sbatch`, into experiments_jepa/pose) and goes through the same filter. The filter
  and the window spacing are not changed.
  Result (job 12484295, 6 min): 74 clips, 263 windows, **66 walking** (from 27 clips) before the outline checks;
  J0 is decided after them.
- **Our clips drawn** (job 12484272, 3 min): all 15 sets, 220 clips each; about 2.5 CPU-minutes per set.

**Result (job 12484912, 36 min on an A10; 2026-11-22): J0 fails, so the judge is not adopted** (stop rule: no
scoring counts without 60 real clips). Version 25 and the ruler plus the user's before/after stay as they were.
- **Real clips:** SAM 2.1's outlines agree with Lab4D's at the unprompted kept frames (IoU median 0.96 to 0.99 per
  clip). Of 263 windows from 74 clips, 66 pass the walking filter and 142 are seen whole, but only **34** are both
  (13 videos; 0074 gives 8, 0075 gives 5). J0 needs 60.
- **The walking filter is also not clean:** a look at 5 of the 34 kept clips found 3 walking side-on and 2 that are
  not walks (0025 at 192: a cat lying down; 0001 at 0: a jump). The 17 original clips were screened by eye for walking;
  the 57 added ones were not, and SuperAnimal's swing rule fires on keypoint noise in lying and jumping cats.
- **For information only (the job ran all steps; none of this counts):** J1 2 of 3 (V1 right, significantly; V2
  wrong way, the head fix scored worse, not significantly; V3 right); J2 4 of 5 kinds (sliding, stiff legs, held
  poses, backwards all worse step by step; bone jitter scored *more* real at each step, perhaps because real outlines
  wobble from frame to frame); J3 fails for every version (the two real halves, 6 and 7 videos, are too few: intervals
  span zero widely); J4 passes (about 2.5 CPU-minutes to draw a candidate, 21 GPU-seconds for its features).
- **GPU use:** about 0.75 GPU-hours today (three jobs), within the 2 approved.
- **What would be needed to try again** (not planned; a new decision): many more real side-on walking clips, screened
  for walking without making the user rate them, which this archive does not have.

## Stopped (2026-11-22)
After the judge result, the remaining generative-model ideas (a video model as a motion source, the ruler as a
teacher, sampling poses instead of averaging) were weighed; none looked likely to be clearly better to the user. The
user said "Stop here". Version 25 stays.
