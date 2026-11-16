#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 6
#SBATCH --mem=48G
#SBATCH --gpus-per-node=1
#SBATCH -J claude-afterb1
#SBATCH -o logs/afterb1-%j.out
# Demo (not a test), after the full behaviour run: the copy's own walks (b1_rollout.py) and their body motion on
# Agent-to-Sim's cat (a2s_cat_export.py). Submit with --dependency=afterok:<behaviour job>.
D=${PETS_ROOT:-/scratch/CLUSTER/$USER/claude-pets}
bash $D/run_b1_rollout.sbatch b128 || { echo "rollout failed"; exit 3; }
type module >/dev/null 2>&1 || source /etc/profile
module load gcc/11.5.0 cuda/12.1.1
export PATH=$D/env-a2s/bin:$PATH PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$D/code/agent2sim/lab4d \
  PYOPENGL_PLATFORM=egl DECIMATE_PY=$D/env-a2s/bin/python
unset PYTHONHOME
cd $D/code/agent2sim/gdmdm || exit 2
python $D/tools/a2s_cat_export.py cat-pikachu-2024-08-v2-compose-ft $D/dl/a2s/cat-pikachu-2024-08-v2-compose-ft-train-L64-S1.pkl \
  $D/experiments_viewer/a2s_b128 30000 $D/experiments_b1/b128/rollouts.json || { echo "walk export failed"; exit 4; }
echo "DONE afterb1 $(date)"
