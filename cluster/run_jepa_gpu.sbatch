#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1
#SBATCH -J claude-jepa
# The video-model judge (PROTOCOL_CAT.md, "A video-model judge"), the GPU part in env-judge:
#   real: SAM 2.1 outlines of the real cat-85 clips -> experiments_jepa/clips/real.npz
#   feat: V-JEPA 2 features of every clip set in experiments_jepa/clips (or the named ones)
#   score: the scores and checks J0 to J3 -> experiments_jepa/score.json
#   sbatch -A ACCOUNT -p a10 -q normal --time=00:40:00 -o logs/jepa-%j.out run_jepa_gpu.sbatch real feat score
D=${PETS_ROOT:-/scratch/CLUSTER/$USER/claude-pets}
type module >/dev/null 2>&1 || source /etc/profile
set -u
export PATH=$D/env-judge/bin:$PATH PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HOME=$D/cache/hf TORCH_HOME=$D/cache/torch OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
unset PYTHONHOME PYTHONPATH
cd $D || exit 2
O=experiments_jepa/clips
mkdir -p $O
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
for step in "$@"; do
  t0=$(date +%s)
  case $step in
    real) python tools/jepa_real.py models/sam2/sam2.1_hiera_small.pt experiments_judge/real \
            experiments_gaitvid/raw/database/processed/JPEGImagesRaw/Full-Resolution experiments_jepa/pose $O/real.npz || exit 3 ;;
    feat) python tools/jepa_feat.py models/vjepa2-vitl-fpc64-256 $(ls $O/*.npz) || exit 4 ;;
    score) python tools/jepa_score.py $O experiments_jepa/score.json || exit 5 ;;
  esac
  echo "STEP $step took $(( $(date +%s) - t0 )) s"
done
echo "DONE jepa $* $(date)"
