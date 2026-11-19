#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 6
#SBATCH --mem=48G
#SBATCH --gpus-per-node=1
#SBATCH -J claude-p0past
#SBATCH -o logs/b1roll-%j.out
# Phase 0, P0-2 (PROTOCOL_CAT.md "Restart, Phase 0"): does the body stage use the past facing? (cluster/p0_past.py)
#   sbatch -A ACCOUNT -p a10 -q normal --time=00:20:00 run_p0_past.sbatch
LOG=${1:-b128}
D=${PETS_ROOT:-/scratch/CLUSTER/$USER/claude-pets}
type module >/dev/null 2>&1 || source /etc/profile
module load gcc/11.5.0 cuda/12.1.1 || { echo 'module load failed'; exit 1; }
set -u
A=$D/code/agent2sim
NAME=cat-pikachu-2024-08-v2-compose-ft
export PATH=$D/env-a2s/bin:$PATH PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
unset PYTHONHOME
export PYTHONPATH=$A/lab4d PYOPENGL_PLATFORM=egl
export TORCH_HOME=$D/cache/torch MPLCONFIGDIR=$D/cache/mpl
cd $A/gdmdm || exit 2
P0_OUT=$D/experiments_p0/past B1_STARTS=$D/data/a2s/motion/$NAME-heldout-L64-S10.pkl \
  python $D/tools/p0_past.py --load_logname $NAME --logname_gd $LOG --eval_batch_size 1 --suffix latest \
  || { echo "p0_past failed"; exit 3; }
echo "DONE p0past $LOG $(date)"
