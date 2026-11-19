#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 6
#SBATCH --mem=32G
#SBATCH --gpus-per-node=1
#SBATCH -J claude-clipid
# Screening the close-up clips (PROTOCOL_CAT.md "Better data"): which cat-85 clips show this cat (clip_identity.py).
#   sbatch -A ACCOUNT -p a10 -q normal --time=00:20:00 -o logs/clipid-%j.out run_clip_identity.sbatch
D=${PETS_ROOT:-/scratch/CLUSTER/$USER/claude-pets}
type module >/dev/null 2>&1 || source /etc/profile
module load gcc/11.5.0 cuda/12.1.1 || { echo 'module load failed'; exit 1; }
set -u
export PATH=$D/env-lab4d/bin:$PATH PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
unset PYTHONHOME PYTHONPATH
export TORCH_HOME=$D/cache/torch MPLCONFIGDIR=$D/cache/mpl PETS_ROOT=$D
cd $D || exit 2
L=$D/code/lab4d/database/processed
FRAMES2=$L/JPEGImages/Full-Resolution python tools/clip_identity.py experiments_judge/real \
  data/cat85/x/database/processed/JPEGImages/Full-Resolution $L/JPEGImages/Full-Resolution/cat-pikachu-0-0000 \
  $L/Annotations/Full-Resolution/cat-pikachu-0-0000 experiments_judge/screen/identity.json || { echo "identity failed"; exit 3; }
echo "DONE clipid $(date)"
