#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 12
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1
#SBATCH -J claude-skin
# Smoothing the skin weights (PROTOCOL_CAT.md), CPU only (the GPU is only how the queue hands out nodes): the walking
# cycle rebuilt on the new weights, then checks S1-S5 and the ruler (S4).
#   sbatch -A ACCOUNT -p a10 -q normal --time=00:40:00 -o logs/skinx-%j.out run_skin_check.sbatch ROUND
D=${PETS_ROOT:-/scratch/CLUSTER/$USER/claude-pets}
type module >/dev/null 2>&1 || source /etc/profile
set -u
export PATH=$D/env-lab4d/bin:$PATH PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
unset PYTHONHOME PYTHONPATH
export MPLCONFIGDIR=$D/cache/mpl OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2
cd $D || exit 2
S=experiments_skin/$1
mkdir -p $S/gait
python tools/gait_cycle.py $S/page experiments_feet/ik2 experiments_feet/floor $S/gait > $S/gait.log 2>&1 || { echo "gait failed"; exit 3; }
( OPENBLAS_NUM_THREADS=1 TARGET=0.08 MAXWIN=220 TRACKS=walk PLANT=walk GAIT=$S/gait/gait_bones.npz \
    WINDOWS=experiments_judge/ours/walk_shown_windows.json \
    python tools/judge_ours.py $S/page experiments_feet/ik2 experiments_feet/floor experiments_head/g15/graft.npz $S/ruler_walks.npz > $S/ruler_walks.log 2>&1
  python tools/gait_score.py experiments_judge/real_pairs.npz experiments_judge/ours/walk_shown.npz $S/ruler_walks.npz $S/s4.json ) &
python tools/skin_check.py experiments_feet/page $S/page experiments_feet/ik2 experiments_gait/g5 $S/gait $S/check &
wait
echo "DONE skin checks $1 $(date)"
