# Pet copy, step S1: the person as a stimulus

Written on 2026-11-15 at the user's request ("Yes, plan it"), after the user set the long-term direction
("interaction of the cat with the environment, and maybe stimuli and response"). Nothing in it has been run.
The read-out rule below is fixed now and must not change once results exist.

## Why this stimulus first

The behaviour model from test B1 (Agent-to-Sim's `gdmdm`, trained tonight as `b128`) already takes one stimulus
as an input: where the person filming has been over the last 0.8 s. Its goal and path models use it; its body
model does not (it is given the person times zero). The model was also trained with that input randomly
switched off, so "no person" is a valid input too (`drop_cam`). No other stimulus (food, toys, sounds, touch) is
in the released data, so the person is the only response the copy can have learned.

## Part 1: something to look at (the main deliverable)

In "At home" on Pikachu's Keep:

1. **Put the person somewhere.** Click the floor to place the person; the page snaps to the nearest of 12
   spots and plays the copy's response from there. The 12 spots are where people really stood while filming:
   the centres of 12 clusters of the recorded person positions (k-means over all 19 videos). For each spot,
   3 starting moments (the cat at different places, from the held-back recordings) x 2 random seeds = 72 walks
   of 34 s (6 rounds of 5.6 s). The person stands still at the spot.
2. **The same person, the real cat and the copy side by side.** For the two held-back recordings (videos the
   copy never trained on), replay the person's real movements and let the copy respond round by round from the
   real cat's first 0.8 s; show the real cat and the copy (a lighter "ghost") in the room together. 4 seeds per
   recording.
3. The copy's walks use Agent-to-Sim's cat and its own body motion, as tonight's walks do. The page says plainly
   that the person's position is the only thing the copy reacts to.

## Part 2: a check that it responds at all (reported without a pass rule)

On the 260 held-back moments from B1: sample 16 goals (where the cat is 5.6 s later) under four conditions,
from the same starting state:

- A1, A2: the real person (two independent sample sets; their difference is the sampling noise);
- B: the person moved to the mirror position through the cat (same distance, opposite side);
- C: no person (`drop_cam`).

Measures, per moment, on the centre of the 16 goals: shift B = |centre(B) - centre(A1)|, shift C likewise,
noise = |centre(A2) - centre(A1)|; also whether the shift under B points toward the person's new side, and how
often the copy ends nearer the person than it started, against how often the real cat did (from the recordings).

Read-out fixed now: the copy "responds to where the person is" if the median shift B is at least twice the
median noise; "barely" if between 1x and 2x; "ignores the person" if below 1x. Same wording for C.

## What it would and would not show

- A response would mean the model's choice of goal depends on where the person stands, in this one home.
- It would not show that the cat reacts to the person. In these videos the person is the one filming and
  usually follows the cat, so the model may have learned "the person goes where the cat is about to be" rather
  than "the cat goes toward (or away from) the person". The data cannot separate the two.
- Nothing about voice, gestures, food or touch; nothing beyond 5.6 s per round.

## Cost and steps

About 45 GPU-minutes on an A10, within the 10 GPU-hours the user allowed for exploring (about 1.5 used so far
including tonight's queued walks). Starts after tonight's walks are on the page, with the user's go-ahead.

1. `cluster/b1_stimulus.py`: Part 2, plus the 12 spots (k-means of recorded person positions).
2. `cluster/b1_rollout.py` gains a person override: a fixed spot, or a recorded person path replayed round by
   round.
3. `cluster/a2s_cat_export.py` converts the new walks' joint angles to body motion (as tonight).
4. Viewer: click-to-place person, ghost of the real cat, labels.

## Changes

**2026-11-15, 12:30, after the user's go-ahead ("Yes, run it"), before any result.** Built and tested end to
end on the 30-epoch trial model (job 12389542, short settings; its numbers are not results and are not
reported). Details fixed in the code, none changing the read-out:

- Part 2 (`cluster/b1_stimulus.py`): A1 and A2 use seeds 1 and 2; B and C use seed 1, so the only difference
  from A1 is the person. "Mirrored" negates the person's whole past 0.8 s horizontally (x and z) relative to
  the cat, keeping the height. Distances are horizontal. The 12 spots are k-means (seed 0) of the person's
  position at the current moment of every training and held-back window.
- Part 1 (`cluster/b1_rollout.py`, modes `spots` and `replay`): the 3 starting moments are the held-back windows
  at 1/4, 1/2 and 3/4 of the list; the person at a spot stands still (their past 0.8 s is the spot); replays run
  in rounds of 5.6 s until the recording ends, the person's past 0.8 s taken from the recording each round.
- Everything for the real model runs as one job (`run_s1.sbatch b128`, job 12389742), queued to start after
  tonight's walks job.

## Result

**2026-11-15, 18:45, real model `b128` (job 12389742).** `experiments_s1/b128/s1_check.json`, `spots.json`.

Part 2, on the 260 held-back moments, 16 goals per condition, horizontal metres:

| measure | value |
|---|---|
| noise: median \|centre(A2) - centre(A1)\| | 0.022 |
| B, person mirrored: median shift | 0.569 (25.6 x noise) |
| C, no person: median shift | 0.462 (20.8 x noise) |
| B: share of moments shifting toward the person's new side | 59% (median 0.04 m toward) |
| copy's goals ending nearer the person than at the start | 41% |
| real cat, same (held-back / training) | 38% / 39% |

**Read-out, by the fixed rule: the copy "responds to where the person is" (B), and "responds" to the
person's absence (C).** Reported alongside, without a rule:

- The noise is tiny because the copy's 16 goals clump together (as seen in B1), so the 2x bar is easy to
  clear. The shifts are large in absolute terms too (about 0.5 m, against 0.78 m goal error in B1).
- The response has no clear direction: moving the person to the other side moves the goal toward the
  person's new side only slightly more often than not, and the copy ends nearer the person about as often
  as the real cat did. So the person changes where it goes, but not simply "toward the person".
- The caveat above stands: the filmer follows the cat.

Part 1 published as version 7 of Pikachu's Keep: 72 spot walks of 34 s, 8 replays beside the real cat
(4 seeds x 2 videos, 106 s and 157 s), plus the 16 free walks. Total GPU for S1: about 23 minutes.

---

# Step S2: the room, a walking person, and calling the cat

Written on 2026-11-15 at 21:35 after the user said "Build a second cat, also explore stimuli response", before
anything in S2 was run. Within the 10 GPU-hours the user allowed for exploring (about 3 used so far). The read-outs
below are fixed now.

## What else the copy can respond to

Besides the person's last 0.8 s (S1), the behaviour model reads the room: a 6.4 x 0.8 x 6.4 m grid of features
around the cat, taken from the room's 3D reconstruction. Its training randomly zeroed that input, so "no room" is
a valid input too (`drop_feat`). It also accepts a goal from outside: Agent-to-Sim's own demo lets a user click
where the cat should go, and the model then picks the route and the body motion.

## Part 1: something to look at

"Call the cat": click near one of 12 places the real cat goes (k-means of every recorded position, in 3D so a
sofa top and the floor under it stay apart); the copy walks there by its own route with its own body motion, then
carries on by itself for two more rounds (17 s in all). 3 starting moments (the same held-back windows as S1) per
place, one seed: 36 walks.

## Part 2: checks (same 260 held-back moments, 16 goals each, as in S1)

Conditions: A1, A2 = as recorded, seeds 1 and 2 (their difference is the sampling noise); N = no room (scene
input zeroed), seed 1; App / Ret = the person walking toward / away from the cat at 0.5 m/s along the line
between them, ending where they really are now, seed 1; Still = the person standing where they are now.

- The room: shift = |centre(N) - centre(A1)| against noise; and placement: the share of goals within 0.25 m (3D)
  of somewhere the real cat has been (any training or held-back moment), with and without the room.
  Read-outs: shift, as S1: "responds" if the median shift is at least 2x the median noise, "barely" at 1-2x,
  "ignores" below. Placement: "keeps goals where the cat goes" if the share drops by at least 10 points without
  the room, "slightly" at 3-10 points, "no effect on placement" below 3.
- The walking person, on moments where the person is 0.8-3 m from the cat: shift = |centre(App) - centre(Ret)|
  against the noise on the same moments, same wording. Reported without a rule: whether an approaching person
  draws the goal toward or away (the component along the cat-to-person line), and the share of moments each way.

## What it would and would not show

- A room effect would mean the copy's choices depend on the space around it in this one home. It cannot separate
  "avoids walls" from "goes to favourite spots"; the placement share only says whether goals stay in places the
  real cat used.
- The walking person shares S1's caveat: in these videos the filmer follows the cat, so a reaction to an
  approaching person may be the model's memory of "the camera moves toward where the cat is going".
- "Call the cat" is a demo of goal-following, not a test: the goal is given.

## Steps

`cluster/b1_room.py` (checks and places), `cluster/b1_rollout.py` mode `call`, `cluster/run_s2.sbatch b128` (one
job, about 30 GPU-minutes; also recomputes body motion for all walks).

## S2 result

**2026-11-15, 21:52, model `b128` (job 12393526, about 25 GPU-minutes including body motion for all 132 walks).**
`experiments_s2/b128/s2_check.json`, `places.json`. Noise (A2 vs A1, median over 260 moments): 0.022 m.

| measure | value | read-out (fixed above) |
|---|---|---|
| no room: median goal shift | 0.119 m (5.4 x noise) | responds |
| goals within 0.25 m of where the real cat has been: with room / without | 89.5% / 86.7% (real goals: 100%) | drop 2.9 points: no effect on placement (just under "slightly") |
| person walking toward vs away (100 moments, person 0.8-3 m away): median shift | 1.06 m (66 x noise on those moments) | responds |
| approaching person: goal toward (+) or away (-) | median -0.67 m; away in 80% of moments | (no rule) |
| person standing still vs as recorded: median shift | 0.32 m | (no rule) |

Reading, without changing the rules: the room input matters, but modestly next to the person (0.12 m against
0.57 m for mirroring the person in S1 and 1.06 m for the person's walking direction here); and the placement
measure is loose (even without the room the copy's goals mostly stay near the cat, and somewhere near the cat is
usually somewhere it has been). The walking-person effect has a clear direction: an approaching person pushes the
goal away, a retreating one draws it along. That is exactly the pattern of the filmer following the cat ("the
camera moved toward the cat" mostly happened when the cat was walking away from it), so it cannot be read as the
cat fleeing.

Part 1 published as version 10 of Pikachu's Keep: 36 "call the cat" walks (12 places x 3 starts, 17 s), with all
earlier walks.
