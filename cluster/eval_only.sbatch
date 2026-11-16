#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=24G
#SBATCH --gpus-per-node=1
#SBATCH -J claude-pets-eval
#SBATCH -o logs/eval-%j.out
# Re-runs pet_eval.py on a finished reconstruction, without training.
# usage (from the pets folder): sbatch -A ACCOUNT -p a10 -q normal --time=00:30:00 eval_only.sbatch score|orbit EXPERIMENT_DIR
WHAT=$1; EXP=$2
D=${PETS_ROOT:-/scratch/CLUSTER/$USER/claude-pets}
type module >/dev/null 2>&1 || source /etc/profile
module load gcc/11.5.0 cuda/12.1.1 || { echo 'module load failed'; exit 1; }
set -u
export PATH=$D/env/bin:$PATH PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
unset PYTHONHOME PYTHONPATH
export TORCH_HOME=$D/cache/torch TRITON_CACHE_DIR=$D/cache/triton MPLCONFIGDIR=$D/cache/mpl
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $D/code/4D-Animal
python $D/tools/pet_eval.py $WHAT $EXP || { echo "$WHAT failed"; exit 4; }
echo "DONE $WHAT $EXP $(date)"
