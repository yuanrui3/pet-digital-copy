#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 6
#SBATCH --mem=48G
#SBATCH --gpus-per-node=1
#SBATCH -J claude-b1roll
#SBATCH -o logs/b1roll-%j.out
# Behaviour demo (not a test): the trained model walks the cat around on its own (cluster/b1_rollout.py).
#   sbatch -A ACCOUNT -p a10 -q normal --time=00:40:00 run_b1_rollout.sbatch LOG     (LOG = b128 for the full model)
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
B1_OUT=$D/experiments_b1/$LOG B1_STARTS=$D/data/a2s/motion/$NAME-heldout-L64-S10.pkl \
  python $D/tools/b1_rollout.py --load_logname $NAME --logname_gd $LOG --eval_batch_size 1 --suffix latest \
  || { echo "rollout failed"; exit 3; }
echo "DONE b1roll $LOG $(date)"
