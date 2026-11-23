#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --gpus-per-node=1
#SBATCH -J claude-sapose
# Real cats' gait from video (PROTOCOL_CAT.md), step 1: the cat-85 clips' raw frames (from the archive already on the
# cluster) and SuperAnimal-Quadruped keypoints on every one (sa_pose.py), with a check sheet.
#   sbatch -A ACCOUNT -p a10 -q normal --time=00:40:00 -o logs/sapose-%j.out run_sa_pose.sbatch CLIP...
D=${PETS_ROOT:-/scratch/CLUSTER/$USER/claude-pets}
type module >/dev/null 2>&1 || source /etc/profile
module load gcc/11.5.0 cuda/12.1.1 || { echo 'module load failed'; exit 1; }
set -u
export PATH=$D/env-lab4d/bin:$PATH PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
unset PYTHONHOME PYTHONPATH
export TORCH_HOME=$D/cache/torch OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
cd $D || exit 2
G=experiments_gaitvid
mkdir -p $G/raw $G/pose
for c in "$@"; do
  [ -d $G/raw/database/processed/JPEGImagesRaw/Full-Resolution/$c ] || \
    unzip -q -o data/cat85/cat-85.zip "database/processed/JPEGImagesRaw/Full-Resolution/$c/*" -d $G/raw
done
SHEET=1 python tools/sa_pose.py models/superanimal_quadruped/pose_model.pth experiments_judge/real \
  $G/raw/database/processed/JPEGImagesRaw/Full-Resolution $G/pose "$@"
echo "DONE sa_pose $(date)"
