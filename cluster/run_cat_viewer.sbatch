#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 6
#SBATCH --mem=32G
#SBATCH --gpus-per-node=1
#SBATCH -J claude-catviewer
#SBATCH -o logs/catviewer-%j.out
# Demo (not a test): export the C1 cat as an animated coloured mesh for the web viewer.
#   sbatch -A ACCOUNT -p a10 -q normal --time=00:40:00 run_cat_viewer.sbatch fg-skel-heldout-r120 [extra flags]
LOG=${1:-fg-skel-heldout-r120}; shift
D=${PETS_ROOT:-/scratch/CLUSTER/$USER/claude-pets}
type module >/dev/null 2>&1 || source /etc/profile
module load gcc/11.5.0 cuda/12.1.1 || { echo 'module load failed'; exit 1; }
set -u
export PATH=$D/env-lab4d/bin:$PATH PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
unset PYTHONHOME PYTHONPATH
export TORCH_HOME=$D/cache/torch MPLCONFIGDIR=$D/cache/mpl DECIMATE_PY=$D/env-a2s/bin/python
cd $D/code/lab4d || exit 2
SEQ=cat-pikachu-0
python $D/tools/cat_viewer_export.py --flagfile=logdir/$SEQ-$LOG/opts.log --load_suffix latest \
  --cat_out $D/experiments_cat/$LOG/viewer "$@" || { echo "viewer export failed"; exit 3; }
echo "DONE catviewer $LOG $(date)"
