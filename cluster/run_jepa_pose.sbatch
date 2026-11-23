#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --gpus-per-node=1
#SBATCH -J claude-jepa-pose
# The video-model judge (PROTOCOL_CAT.md, "A video-model judge"): more cat-85 clips for the real side, as step 1 of
# "Real cats' gait from video" did for 17: raw frames from the archive, SuperAnimal keypoints (for the walking filter)
# in experiments_jepa/pose, with the 17 clips' keypoints linked in.
#   sbatch -A ACCOUNT -p a10 -q normal --time=00:40:00 -o logs/jepa-pose-%j.out run_jepa_pose.sbatch CLIP...
D=${PETS_ROOT:-/scratch/CLUSTER/$USER/claude-pets}
type module >/dev/null 2>&1 || source /etc/profile
module load gcc/11.5.0 cuda/12.1.1 || { echo 'module load failed'; exit 1; }
set -u
export PATH=$D/env-lab4d/bin:$PATH PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
unset PYTHONHOME PYTHONPATH
export TORCH_HOME=$D/cache/torch OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
cd $D || exit 2
G=experiments_gaitvid; P=experiments_jepa/pose
mkdir -p $G/raw $P
for c in "$@"; do
  [ -d $G/raw/database/processed/JPEGImagesRaw/Full-Resolution/$c ] || \
    unzip -q -o data/cat85/cat-85.zip "database/processed/JPEGImagesRaw/Full-Resolution/$c/*" -d $G/raw
done
SHEET=1 python tools/sa_pose.py models/superanimal_quadruped/pose_model.pth experiments_judge/real \
  $G/raw/database/processed/JPEGImagesRaw/Full-Resolution $P "$@"
for f in $G/pose/cat-*.npz; do [ -e $P/$(basename $f) ] || ln -s ../../$f $P/; done
echo "DONE sa_pose $(date)"
