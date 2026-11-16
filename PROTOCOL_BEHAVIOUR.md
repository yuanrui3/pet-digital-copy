# Pet copy, behaviour test B1: does a copy act like this cat in a recording it never saw?

Written on 2026-11-14 at the user's request ("Yes, plan it"); approved the same evening ("Yes, run it"),
before anything was downloaded or run. The pass rule below is now fixed and must not change once results
exist.
Companion to `PROTOCOL.md` (dogs, look) and `PROTOCOL_CAT.md` (cat, look): same structure, same idea of
a "replay what was recorded" baseline.

## Question

The novel's Keep copies behaviour, not only looks: "pose tracking, behaviour cloning", and whatever the
cat never did on camera comes from a prior. The real question underneath: can released, open code learn
one cat's behaviour from a month of casual video well enough that, in a recording it never saw, it
predicts what the cat does next better than simply replaying the most similar moment from the cat's
past recordings? Replay is what the novel says the copy really is ("the cat's younger, recorded
self"), and it is the behaviour version of the look tests' "show the nearest photo" baseline.

Not tested here: whether the copy is this cat rather than an average cat (no other cats' behaviour data
in this form is public; see "Not covered"), sound, and the diary.

## Method and data

| What | Source | Terms |
| --- | --- | --- |
| Behaviour model | Agent-to-Sim (ICLR 2025), `github.com/facebookresearch/agent2sim` at commit `1952ed1` (2024-10-16, archived), folder `gdmdm`: three diffusion models (goal, path, full-body motion) conditioned on the cat's past 0.8 s, the room and the observer (the person filming) | CC BY-NC 4.0 |
| Behaviour data | the authors' 4D reconstruction of their cat (the same cat as the look test) from 23 phone videos over about a month, at 10 frames per second: `cat-pikachu-2024-08-v2-compose-ft-train-L64-S1.pkl` (968,623,155 bytes; every 6.4 s window of 22 videos; the authors' own test video, number 16, is not in it) | CC BY-NC 4.0 (repository licence) |
| Room and observer | the same reconstruction's checkpoint, `log-cat-pikachu-2024-08-v2-compose-ft.zip` (350,283,743 bytes) | CC BY-NC 4.0 |
| Code base | Lab4D's `lab4dgs` branch at the commit Agent-to-Sim pins (`334c1b4`), for its scene reader | MIT |

Settings are the authors' (see Changes for which of their two descriptions): 16 samples per prediction,
12 trials. Nothing tuned.

## What is held back

The training file is a list of overlapping 64-frame windows (stride 1), video after video, without video
labels. Video boundaries are recovered where a window is not the previous window shifted by one frame.
Held back by rule, decided before looking at the data: the last 2 of the 22 videos in the file's order
(about 9% of the windows, like the paper's 1 of 23). All windows of those videos go to testing; none to
training. Training windows never overlap testing windows, because they come from different videos.

What leaks, as in the look tests: the 4D reconstruction (the cat's trajectory, pose and the room) was
built by the authors from all 23 videos at once, so the held-back videos' pixels helped shape the room
and the body model. Their behaviour (what the cat did next) never enters the behaviour model's training.

## Measures (the paper's, `gdmdm/eval.py`)

For each held-back window, given the first 0.8 s, the room and the observer, predict the next 5.6 s:

- goal: where the cat is at the end (minDE, metres);
- path: its route (minADE, metres; conditioned on the true goal, as in the paper);
- orientation (radians) and joint angles (radians; conditioned on the true path).

"min" = the best of 16 sampled predictions, averaged over windows, repeated over 12 trials.

## Baselines (each also produces 16 candidates, so "best of 16" is fair to them)

1. **Replay** (the one that decides): the 16 training windows whose first 0.8 s are most similar to the
   held-back window's (the cat's past path in its own frame, plus the observer's position relative to the
   cat; plain Euclidean distance on those numbers), and what the cat did next in each, laid onto the
   held-back window's starting position.
2. **Location prior** (the paper's own baseline, goal only): goals sampled from where the cat ends up in
   the training windows.
3. **Stay put**: the cat stays where it is, body unchanged (one candidate, repeated).

## Pass rule

- The copy passes if, over the held-back windows, it beats Replay on both goal minDE and path minADE
  (mean over the 12 trials).
- Reported without a rule, whatever they show: orientation and joint angles against Replay; all three
  baselines on every measure; the share of held-back windows where the copy's goal is better than
  Replay's; the paper's own numbers for comparison (their test video differs from ours, so only roughly).

## What a pass would and would not mean

- It would mean the learned copy predicts this cat's next few seconds, in its own home, better than
  replaying its most similar recorded moments: the copy has generalised beyond its recordings.
- It would not show that the copy behaves like this cat rather than like a typical cat, nor anything
  outside the one room the videos were filmed in, nor anything beyond 5.6 s.

## Not covered: "this cat or an average cat?"

The novel's question of how far the copy leans toward the average cat needs behaviour data from other
cats in the same form. None is public as far as I found: Agent-to-Sim released only this cat, one bunny
scene and one human. A later test could build it from Lab4D's 85-video cat set (`cat-85`, 41.7 GB, many
different cats, internet videos), but those are short clips in many rooms, so the comparison would be
loose.

## Cost and steps

1. Download the two files above to the cluster (about 1.3 GB). Check that the 22 videos are recovered.
2. A setup script the user runs once (Agent-to-Sim plus Lab4D's `lab4dgs` branch, `networkx==2.5`),
   added to the existing Lab4D environment or a copy of it.
3. A short trial to measure speed. The paper's training is 10 hours on an A100; my estimate for an A10
   is 25-30 GPU-hours, to be confirmed by the trial. If the trial shows more than 36 hours, I stop and ask.
4. One full training run, scoring with the paper's evaluation plus the three baselines, and a few
   side-by-side clips of predicted and real behaviour in the room.

## Changes

**2026-11-14, 23:00, after approval, before any training or scoring.**

- Downloaded and checked: both files above, sizes as listed. The checkpoint folder carries the authors'
  cached data summary (`tmp/cat-pikachu-2024-08-v2-compose-ft.pkl`), so the room model loads without the
  original videos (not public); its per-video export folders are empty, so the room's mesh and features
  are rebuilt from the checkpoint by the code itself on the first GPU run. No per-video motion files are
  included, so the authors' own test video cannot be scored; the split above stands.
- Software: Agent-to-Sim pins Lab4D's `lab4dgs` branch, whose quaternion CUDA extension differs from the
  one the cat test uses, so this test gets its own environment (`env-a2s`, built by `cluster/setup_a2s.sh`,
  which the user runs): Python 3.10, PyTorch 2.4.1 / CUDA 12.1, the extension compiled as C++17, and the
  packages the training imports (traced statically from `gdmdm/train.py`): open3d, viser, pyrender,
  recordclass, celluloid, geomloss, point-cloud-utils, torchcubicspline, urdfpy with networkx 2.5 (as the
  README says), plus the `ppr-diffphys` submodule for the cat's skeleton file. If urdfpy trips on NumPy
  aliases removed in 1.24, they are mapped to their 64-bit types; nothing else in third-party code is
  edited.
- Scoring: the release contains the paper's metric (`gdmdm/eval.py`, `eval_ADE`) but not the script that
  produced its Table 4 for the full model. I write that part (sampling 16 goals, paths given the true goal,
  and body motion given the true path, through the model's own sampling code), and use `eval_ADE` for
  every method, baselines included. Correction, same evening: the release does contain that script after
  all (`gdmdm/demo.py`: `run_eval` then `eval_all` with 192 samples = 16 x 12 trials). Scoring uses its
  calls (`Agent.update_goal`, `update_waypoint`, `update_fullbody`, `eval_all`), looped over the held-back
  windows in small batches instead of the two windows the demo picks.

**2026-11-14, 23:20, after splitting the file, before any training.**

- The file holds 19 videos, not the 22 the paper's split suggested (`cluster/b1_split.py`, `split.json`:
  10,377 windows; videos of 24 to 162 s). The rule held back "the last 2 videos in the file's order"; those
  happen to be the two longest (162 s and 110 s), so 25% of the windows are held back, not the 9% the
  draft estimated. The rule is kept as written. Training: 17 videos, 7,788 windows.
- Held-back windows are scored every 10 frames (1 s), not at every frame: 260 windows. Neighbouring
  windows one frame apart are near copies of each other; this keeps the scoring time small and changes
  nothing else.
- Training settings: the README's command for exactly this data (`bash train.sh
  cat-pikachu-2024-08-v2-compose-ft b128 128 1`: batch 128, 4,001 epochs) rather than the paper's prose
  (120k steps of batch 1024), because the released code documents the former. Everything else at the
  code's defaults.
- Replay's matching numbers, made concrete before any scoring: the cat's past 0.8 s path and the
  observer's past positions (both relative to the cat's position now), where the cat is in the room now,
  and which way it faces now (its 3 x 3 orientation). For the path measure, which is given the true goal,
  replay also matches on the goal; for the body measure, given the true path, it matches on the path. So
  each replay candidate answers the same question the copy answers.

**2026-11-15, 01:50, after the first timing trial (job 12386048) stopped at a missing file, before any
result.** The behaviour model's room reader (`BGField(..., use_default_mesh=True)`, as the code calls it)
builds the room's feature grid from the checkpoint (that worked) and then loads the authors' phone scan
of the same room as its reference mesh, `database/polycam/cat-pikachu/Oct5at10-49AM-poly/raw.ply`. It is
in `polycam_all.zip` (1,494,196,201 bytes), which the README lists as training data; only that room's
folder was unpacked. Trial resubmitted unchanged (job 12386844).

**2026-11-15, 02:30, after the timing trial (job 12386844, 30 epochs) ran end to end.** About 13 s per
epoch, so the full 4,001 epochs are about 14-15 GPU-hours, under the 36-hour limit; scoring 260
windows took 20 minutes. The trial's scores are not results (under 1% of the training) and are not
reported as such. Full run submitted (job number in `logs/`).

## Result

**2026-11-15, 18:11, full run (job 12387528, 4,001 epochs), scored on the 260 held-back windows,
12 trials of 16 samples.** `experiments_b1/b128/b1_results.json`, `b1_examples.png`.

| method | goal minDE (m) | path minADE (m) | orientation (rad) | joints (rad) |
|---|---|---|---|---|
| copy | 0.781 | 0.164 | 0.470 | 0.362 |
| Replay | 0.786 | 0.347 | 0.674 | 0.421 |
| Location prior | 0.752 | - | - | - |
| Stay put | 0.988 | 0.545 | 0.820 | 0.298 |
| paper, Table 4 (other test video) | 0.448 | 0.234 | 0.55 | 0.237 |

**By the rule: passes** (beats Replay on goal and on path). Reported without a rule:

- The goal win is a near tie: 0.005 m, about one trial-to-trial standard deviation (0.0047 m); the copy's
  goal is better than Replay's in 52% of windows. The location prior (goals drawn from where the cat
  usually ends up) beats both on goal.
- The path win is large: less than half Replay's error. Orientation is also clearly better.
- On joint angles, Stay put (body unchanged) beats both the copy and Replay.
- In the 6 example windows, the copy's 16 goals usually sit in one tight clump, so when the clump is in the
  wrong place all 16 are wrong; Replay's and the location prior's candidates spread wider.

## B2: better goal choice (plan, written 2026-11-16 at 01:00, before anything was run)

The user picked "Better goal choice" from three directions offered after B1. Why: in B1 the copy's goal
(minDE 0.781 m) barely beat Replay (0.786) and lost to the location prior (0.752); its 16 goals usually sit
in one tight clump; and in S2, removing the room barely changed whether goals land where the cat has been.

**Change (at walk time; no retraining).** "Copy + places": each of the 16 goals is either the copy's own
sampled goal or a place drawn from where the cat ended up in the training windows (the location prior's
10 cm cells, weighted by count), restricted to places within reach: no farther from where the cat is now
than the 99th percentile of goal distances in the training windows. Fixed now, not tuned on held-back
windows: half and half (8 of 16 each; w = 0.5), the 99th percentile, the same cells as B1.

**Measures, on the same held-back windows, 12 trials x 16 goals.**
- goal minDE (B1's measure: the best of 16, metres);
- energy score (a proper score for a set of guesses: mean distance from the guesses to the truth minus half
  the mean distance between guesses; lower is better). Unlike best-of-16, it does not reward scattering
  guesses for its own sake;
- one-goal error: the mean distance of a single drawn goal from the truth (what one round of a walk uses).

**Compared:** the old copy (w = 0; must reproduce B1's 0.781 within one trial-to-trial SD, as a check), the
mix (w = 0.5), places within reach alone (w = 1), B1's location prior, Replay. Also w = 0.25 and 0.75, as
context only.

**Rule.** The mix passes if its goal minDE is lower than both the location prior's and Replay's, and its
energy score is lower than the old copy's. Fixed wording besides: if places within reach alone beat the mix
on both minDE and energy score, the result says "the copy's own goals add nothing beyond favourite places".

**Then.** If the mix passes, the 16 free walks on the page are made again with it (each round: with chance
0.5 a place within reach, otherwise the copy's own goal), and the note says so. The S1 and S2 walks keep the
old goal choice, since their results were measured with it. One GPU job, goals only (about 30 minutes).

## B2 result (2026-11-16, 08:05; job 12396014)

The first run (job 12395993) crashed before scoring: Replay's nearest-moment lookup had its arguments swapped.
Fixed (`nearest(features(test), features(train))`, as in B1) and run again unchanged otherwise. 260 held-back
windows, 12 trials x 16 goals; reach = 3.05 m (99th percentile of training goal distances); every window had
places in reach.

| goals | minDE (m) | trial SD | energy score | one-goal error (m) |
|---|---|---|---|---|
| old copy (w = 0) | 0.782 | 0.0035 | 0.866 | 0.946 |
| w = 0.25 (context) | 0.570 | 0.0128 | 0.715 | 1.185 |
| **mix (w = 0.5, the rule)** | **0.497** | 0.0104 | **0.674** | 1.423 |
| w = 0.75 (context) | 0.458 | 0.0102 | 0.742 | 1.660 |
| places in reach alone (w = 1) | 0.622 | 0.0170 | 0.924 | 1.900 |
| B1's location prior | 0.757 | 0.0180 | 1.370 | 2.817 |
| Replay | 0.786 | - | 0.879 | 1.101 |

Check: the old copy reproduces B1 (0.782 against 0.781; SD 0.0035).

**By the rule: passes.** The mix's minDE (0.497) is below the location prior's (0.757) and Replay's
(0.786), and its energy score (0.674) is below the old copy's (0.866). Places alone do worse than the mix on
both, so the copy's own goals do add something. Reported without a rule:

- The mix's best-of-16 beats the location prior in 73% of windows and the old copy in 67%.
- The energy score is best at w = 0.5 of the five weights; minDE keeps falling to w = 0.75, as best-of-16
  rewards spread.
- One-goal error gets worse (1.42 m against 0.95 m): a single drawn goal is on average farther from where the
  cat went, because half are spread-out places. The gain is in covering where the cat might go, not in any
  one guess.
- Walks (16 free walks made again with w = 0.5; 102 of 192 rounds went to a place), horizontal movement over
  5.6 s, against the 19 real videos: real cat median 0.34 m (16% of stretches over 1.5 m); old walks 0.03 m
  (1%); new walks 0.68 m (27%). Speed over 0.5 s, 99th percentile: real 0.67 m/s, new walks 0.62 m/s (max 0.86).
  So the old copy barely moved and the new one moves somewhat more than the real cat, at real-cat speeds.
  Rounds aimed at a place moved a median 1.38 m; the copy's own rounds 0.13 m.

On the page as version 12 of Pikachu's Keep: the 16 free walks, a legend that says when the copy heads for
one of the cat's usual spots, and a second paragraph in "The copy on its own". S1 and S2 walks unchanged.
Scripts: `cluster/b2_goal.py`, `cluster/b1_rollout.py` (B2_PRIOR, B2_W), `cluster/run_b2.sbatch`.

## B3: matching the walks' movement to the real cat (plan, written 2026-11-16 at 08:40, before running)

The user said "Yes, tune it" to: pick how often the walks head for one of the cat's usual spots (B2's w) so
that the walks move about as much as the real cat. This is a choice for the page's walks, made on training
data only; B2's scored result stays as it is.

**What the real cat does (17 training videos, from the page's recorded paths; videos 18-19 are the held-back
pair).** Horizontal movement over 5.6 s, stretches starting every 1 s (800 stretches): mean 0.541 m,
median 0.084 m, 35.0% over 0.5 m, 12.4% over 1.5 m. The cat is still most of the time and moves a lot when it
moves, so the median mostly says how often it rests; the target is the **mean, 0.541 m**. (While choosing
this I also printed the same numbers for the held-back pair, a much busier stretch of the cat's life: median
0.968 m, mean 0.985 m. They are not used.)

**Walks to choose from.** w = 0, 0.05, ..., 0.5 (11 values); for each, 16 walks of 12 rounds (8 starting
moments spread over the training windows x 2 seeds), measured the same way (stretches every 1 s, the walk's
own frames only).

**Choice.** The w whose walks' mean movement is closest to 0.541 m; but if a smaller w comes within 0.027 m
(5% of the target) of that best distance, the smallest such w (closer to the copy's own choice). Reported beside it: share over
0.5 m and over 1.5 m against the real 35.0% and 12.4%.

**Then.** The page's 16 free walks are made again from the same held-back starts with the chosen w. B2's
scores at that w on the held-back windows are reported as context; if its goal minDE is not below both the
location prior's and Replay's, the result says so.

## B3 result (2026-11-16, 09:31; job 12396085, about 1 GPU-hour)

Walks from training moments (16 per w), mean horizontal movement over 5.6 s against the real cat's 0.541 m in
the 17 training videos:

| w | mean (m) | over 0.5 m | over 1.5 m | rounds to a place |
|---|---|---|---|---|
| 0 | 0.232 | 17.7% | 1.7% | 0/192 |
| 0.05 | 0.313 | 25.0% | 3.3% | 11 |
| 0.1 | 0.361 | 28.5% | 4.7% | 18 |
| 0.15 | 0.427 | 32.9% | 7.7% | 31 |
| 0.2 | 0.489 | 37.8% | 8.6% | 39 |
| **0.25 (picked)** | **0.559** | 42.0% | 10.7% | 47 |
| 0.3 | 0.645 | 46.0% | 14.7% | 60 |
| 0.35 | 0.752 | 52.2% | 18.2% | 73 |
| 0.4 | 0.859 | 59.4% | 21.7% | 81 |
| 0.45 | 0.868 | 60.8% | 21.2% | 87 |
| 0.5 | 0.864 | 59.3% | 21.5% | 98 |
| real cat (training videos) | 0.541 | 35.0% | 12.4% | - |

**Picked: w = 0.25** (0.018 m from the target; w = 0.2 is 0.052 m away, outside the 0.027 m tie margin).
The real shares fall between those at w = 0.2 and 0.25. Movement levels off above w = 0.4.

At w = 0.25 on B2's held-back windows (context; computed with 4 of 16 goals from places): goal minDE 0.570,
below the location prior's 0.757 and Replay's 0.786; energy score 0.715, below the old copy's 0.866. So it
still beats both baselines and the old copy, a little less than w = 0.5 on both measures.

Page (version 13 of Pikachu's Keep): the 16 free walks made again from the same held-back starts with
w = 0.25 (50 of 192 rounds went to a place; mean movement 0.589 m on these starts, against 0.817 m at w = 0.5),
and the note "The copy on its own" updated. S1 and S2 walks unchanged. Scripts: `cluster/b3_pick.py`,
`cluster/run_b3.sbatch`.
