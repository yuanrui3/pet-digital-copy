#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 12
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1
#SBATCH -J claude-judge
# The judge without ratings (PROTOCOL_CAT.md "Building the judge"): our cat drawn and measured in its versions
# (judge_ours.py, CPU, side by side), and the two reconstructions posed in their own videos' cameras
# (judge_cam_export.py, GPU, for check T2). The login node was too slow for the CPU part.
#   sbatch -A ACCOUNT -p a10 -q normal --time=00:45:00 -o logs/judge-%j.out run_judge.sbatch
D=${PETS_ROOT:-/scratch/CLUSTER/$USER/claude-pets}
type module >/dev/null 2>&1 || source /etc/profile
module load gcc/11.5.0 cuda/12.1.1 || { echo 'module load failed'; exit 1; }
set -u
export PATH=$D/env-lab4d/bin:$PATH PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
unset PYTHONHOME PYTHONPATH
export TORCH_HOME=$D/cache/torch MPLCONFIGDIR=$D/cache/mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
cd $D || exit 2
O=experiments_judge/ours
A="experiments_feet/page experiments_feet/ik2 experiments_feet/floor experiments_head/g15/graft.npz"
export TARGET=0.08 MAXWIN=220
run() { python tools/judge_ours.py $A $O/$1.npz > $O/$1.log 2>&1; }
( TRACKS=walk PLANT=walk run walk_shown
  W=$O/walk_shown_windows.json
  TRACKS=walk PLANT=none WINDOWS=$W run walk_none &
  TRACKS=walk PLANT=walk SLIDE_MM=16 WINDOWS=$W run walk_slide16 &
  TRACKS=walk PLANT=walk SLIDE_MM=32 WINDOWS=$W run walk_slide32 &
  TRACKS=walk PLANT=walk BLUR=0 WINDOWS=$W run walk_shown_blur0 &
  TRACKS=walk PLANT=none BLUR=0 WINDOWS=$W run walk_none_blur0 &
  TRACKS=walk PLANT=walk BLUR=0.06 WINDOWS=$W run walk_shown_blur6 &
  TRACKS=walk PLANT=none BLUR=0.06 WINDOWS=$W run walk_none_blur6 &
  wait ) &
( TRACKS=rec PLANT=walk run rec_shown
  TRACKS=rec PLANT=both WINDOWS=$O/rec_shown_windows.json run rec_planted ) &
( cd $D/code/lab4d
  UV_NPZ=$D/experiments_cat/fg-skel-heldout-r120/viewer_tex/c1_uv.npz python $D/tools/judge_cam_export.py \
    --flagfile=logdir/cat-pikachu-0-fg-skel-heldout-r120/opts.log --load_suffix latest --cat_out $D/experiments_judge/cam_pika
  UV_NPZ=$D/experiments_cat2/cat85-0068-fg-skel-r60/uv.npz python $D/tools/judge_cam_export.py \
    --flagfile=logdir/cat85-0068-fg-skel-r60/opts.log --load_suffix latest --cat_out $D/experiments_judge/cam_grey ) &
wait
for f in $O/*.log; do echo "== $f: $(tail -1 $f)"; done
echo "DONE judge $(date)"
