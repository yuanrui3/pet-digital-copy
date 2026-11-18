#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH --gpus-per-node=1
#SBATCH -J claude-home2
#SBATCH -o logs/home2-%j.out
# Second cat at home, retarget (PROTOCOL_CAT.md): export the grey and white cat's own skeleton and skinning.
#   sbatch -A ACCOUNT -p a10 -q normal --time=00:20:00 run_home2.sbatch   (then home2_retarget.py on the login node)
SEQ=${1:-cat85-0068}; LOG=${2:-fg-skel-r60}
D=${PETS_ROOT:-/scratch/CLUSTER/$USER/claude-pets}
type module >/dev/null 2>&1 || source /etc/profile
module load gcc/11.5.0 cuda/12.1.1 || { echo 'module load failed'; exit 1; }
set -u
export PATH=$D/env-lab4d/bin:$PATH PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
unset PYTHONHOME PYTHONPATH
export TORCH_HOME=$D/cache/torch MPLCONFIGDIR=$D/cache/mpl
cd $D/code/lab4d || exit 2
# UV and OUT may be set by the caller (the whole Body tab cat at home: its c1_uv.npz, experiments_home3/skel)
UV_NPZ=${UV:-$D/experiments_cat2/$SEQ-$LOG/uv.npz} python $D/tools/home2_skel.py --flagfile=logdir/$SEQ-$LOG/opts.log \
  --load_suffix latest --cat_out ${OUT:-$D/experiments_home2/skel} || { echo "skeleton export failed"; exit 3; }
echo "DONE home2 $(date)"
