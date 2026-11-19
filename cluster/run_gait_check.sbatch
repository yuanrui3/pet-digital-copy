#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 12
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1
#SBATCH -J claude-gait
# The walking cycle's checks (PROTOCOL_CAT.md "A built-in walking cycle on the copy's walks"), CPU only (the GPU is
# only how the queue hands out nodes): the ruler on the free walks with the cycle (G1), paws and skin (G2, G3).
#   sbatch -A ACCOUNT -p a10 -q normal --time=00:40:00 -o logs/gaitx-%j.out run_gait_check.sbatch GAIT_NAME
D=${PETS_ROOT:-/scratch/CLUSTER/$USER/claude-pets}
type module >/dev/null 2>&1 || source /etc/profile
set -u
export PATH=$D/env-lab4d/bin:$PATH PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
unset PYTHONHOME PYTHONPATH
export MPLCONFIGDIR=$D/cache/mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
cd $D || exit 2
G=experiments_gait/$1
A="experiments_feet/page experiments_feet/ik2 experiments_feet/floor experiments_head/g15/graft.npz"
( TARGET=0.08 MAXWIN=220 TRACKS=walk PLANT=walk GAIT=$G/gait_bones.npz WINDOWS=experiments_judge/ours/walk_shown_windows.json \
    python tools/judge_ours.py $A $G/ruler_walks.npz > $G/ruler_walks.log 2>&1
  python tools/gait_score.py experiments_judge/real_pairs.npz experiments_judge/ours/walk_shown.npz $G/ruler_walks.npz $G/g1.json ) &
OPENBLAS_NUM_THREADS=4 python tools/gait_check.py experiments_feet/page experiments_feet/ik2 experiments_feet/floor $G $G/check &
wait
echo "DONE gait checks $1 $(date)"
