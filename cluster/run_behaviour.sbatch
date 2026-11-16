#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 6
#SBATCH --mem=48G
#SBATCH --gpus-per-node=1
#SBATCH -J claude-b1
#SBATCH -o logs/b1-%j.out
# Behaviour test B1 (work/pets/PROTOCOL_BEHAVIOUR.md): train Agent-to-Sim's behaviour model on the 17 training videos,
# then score it and the baselines on the 2 held-back videos. usage (from the pets folder):
#   sbatch -A ACCOUNT -p a10 -q normal --time=... run_behaviour.sbatch EPOCHS     (4001 = the full run)
EPOCHS=${1:-4001}
D=${PETS_ROOT:-/scratch/CLUSTER/$USER/claude-pets}
type module >/dev/null 2>&1 || source /etc/profile
module load gcc/11.5.0 cuda/12.1.1 || { echo 'module load failed'; exit 1; }
set -u
A=$D/code/agent2sim
NAME=cat-pikachu-2024-08-v2-compose-ft
LOG=b128; [ "$EPOCHS" = 4001 ] || LOG=b128-e$EPOCHS
export PATH=$D/env-a2s/bin:$PATH PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
unset PYTHONHOME
export PYTHONPATH=$A/lab4d PYOPENGL_PLATFORM=egl
export TORCH_HOME=$D/cache/torch MPLCONFIGDIR=$D/cache/mpl
# the training file is the 17-video split, under the name the code expects
ln -sf $D/data/a2s/motion/$NAME-train-L64-S1.pkl $A/lab4d/database/motion/$NAME-train-L64-S1.pkl
cd $A/gdmdm || exit 2
echo "epochs $EPOCHS, log $LOG, $(hostname) $(date)"
torchrun --nproc_per_node 1 --nnodes 1 --rdzv_backend c10d --rdzv_endpoint localhost:0 train.py --ngpu 1 \
  --num_epochs $EPOCHS --load_logname $NAME --logname_gd $LOG --train_batch_size 128 || { echo "training failed"; exit 3; }
echo "trained $(date)"
OUT=$D/experiments_b1/$LOG
mkdir -p $OUT
B1_HELDOUT=$D/data/a2s/motion/$NAME-heldout-L64-S10.pkl B1_OUT=$OUT python $D/tools/b1_eval.py \
  --load_logname $NAME --logname_gd $LOG --eval_batch_size 192 --eval_ntrial 12 --suffix latest \
  || { echo "scoring failed"; exit 4; }
cp $D/data/a2s/motion/split.json $OUT/
echo "DONE b1 $LOG $(date)"
