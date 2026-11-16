#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 6
#SBATCH --mem=48G
#SBATCH --gpus-per-node=1
#SBATCH -J claude-catphoto
#SBATCH -o logs/catphoto-%j.out
# Demo (not a test): paint the real video frames onto the C1 cat's UV-mapped mesh.
#   sbatch -A ACCOUNT -p a10 -q normal --time=00:30:00 run_cat_photo.sbatch fg-skel-heldout-r120
LOG=${1:-fg-skel-heldout-r120}
D=${PETS_ROOT:-/scratch/CLUSTER/$USER/claude-pets}
type module >/dev/null 2>&1 || source /etc/profile
module load gcc/11.5.0 cuda/12.1.1 || { echo 'module load failed'; exit 1; }
set -u
export PATH=$D/env-lab4d/bin:$PATH PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
unset PYTHONHOME PYTHONPATH
export TORCH_HOME=$D/cache/torch MPLCONFIGDIR=$D/cache/mpl
export UV_NPZ=$D/experiments_cat/$LOG/viewer_tex/c1_uv.npz NET_TEX=$D/experiments_cat/$LOG/viewer/cat_tex_fixed.jpg
cd $D/code/lab4d || exit 2
SEQ=cat-pikachu-0
python $D/tools/cat_photo_texture.py --flagfile=logdir/$SEQ-$LOG/opts.log --load_suffix latest \
  --cat_out $D/experiments_cat/$LOG/viewer_photo || { echo "photo texture failed"; exit 3; }
echo "DONE catphoto $LOG $(date)"
