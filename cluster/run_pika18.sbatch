#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1
#SBATCH -J claude-pika18
# The rebuild from this cat's close-up clips (PROTOCOL_CAT.md "The rebuild from this cat's close-up clips"): Lab4D on
# 18 clips of the cat as one cat (database/configs/pika18.config), the settings that built the Body tab cat.
#   trial: sbatch -A ACCOUNT -p a10 -q normal --time=00:30:00 -o logs/pika18-%j.out run_pika18.sbatch 4
#   full:  sbatch -A ACCOUNT -p a10 -q normal --time=10:00:00 -o logs/pika18-%j.out run_pika18.sbatch 160
ROUNDS=${1:-160}
D=${PETS_ROOT:-/scratch/CLUSTER/$USER/claude-pets}
type module >/dev/null 2>&1 || source /etc/profile
module load gcc/11.5.0 cuda/12.1.1 || { echo 'module load failed'; exit 1; }
set -u
export PATH=$D/env-lab4d/bin:$PATH PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
unset PYTHONHOME PYTHONPATH
export TORCH_HOME=$D/cache/torch MPLCONFIGDIR=$D/cache/mpl
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $D/code/lab4d || exit 2
SEQ=pika18; LOG=fg-skel-r$ROUNDS
echo "pika18 rounds $ROUNDS $(hostname) $(date)"
SAVE=10; [ $((ROUNDS % 10)) -eq 0 ] || SAVE=1
torchrun --nproc_per_node 1 --nnodes 1 --rdzv_backend c10d --rdzv_endpoint localhost:0 lab4d/train.py --ngpu 1 \
  --seqname $SEQ --logname $LOG --fg_motion skel-quad --num_rounds $ROUNDS --save_freq $SAVE \
  || { echo "training failed"; exit 3; }
echo "DONE pika18 $SEQ $LOG $(date)"
