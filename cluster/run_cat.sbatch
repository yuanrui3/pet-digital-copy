#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 6
#SBATCH --mem=32G
#SBATCH --gpus-per-node=1
#SBATCH -J claude-cat
#SBATCH -o logs/cat-%j.out
# Cat test C1 (work/pets/PROTOCOL_CAT.md): Lab4D on cat-pikachu-0 with the middle quarter's colour held back,
# then scoring and a turntable. usage (from the pets folder):
#   sbatch -A ACCOUNT -p a10 -q normal --time=... run_cat.sbatch ROUNDS     (120 = the full run)
ROUNDS=${1:-120}
D=${PETS_ROOT:-/scratch/CLUSTER/$USER/claude-pets}
type module >/dev/null 2>&1 || source /etc/profile
module load gcc/11.5.0 cuda/12.1.1 || { echo 'module load failed'; exit 1; }
set -u
export PATH=$D/env-lab4d/bin:$PATH PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
unset PYTHONHOME PYTHONPATH
export TORCH_HOME=$D/cache/torch MPLCONFIGDIR=$D/cache/mpl
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $D/code/lab4d || exit 2
SEQ=cat-pikachu-0
LOG=fg-skel-heldout-r$ROUNDS
# the held-back block, fixed by rule: filtered frames floor(0.375 N) .. floor(0.625 N) - 1
N=$(ls database/processed/JPEGImages/Full-Resolution/$SEQ-0000/*.jpg | wc -l)
A=$(( N * 375 / 1000 )); B=$(( N * 625 / 1000 ))
echo "filtered frames $N, held back $A..$((B - 1)), rounds $ROUNDS, log $LOG, $(hostname) $(date)"
SAVE=10; [ $((ROUNDS % 10)) -eq 0 ] || SAVE=1
torchrun --nproc_per_node 1 --nnodes 1 --rdzv_backend c10d --rdzv_endpoint localhost:0 lab4d/train.py --ngpu 1 \
  --seqname $SEQ --logname $LOG --fg_motion skel-quad --num_rounds $ROUNDS --save_freq $SAVE \
  --heldout_frames $A,$B || { echo "training failed"; exit 3; }
OUT=$D/experiments_cat/$LOG
mkdir -p $OUT
python $D/tools/cat_eval.py --flagfile=logdir/$SEQ-$LOG/opts.log --load_suffix latest --eval_res 256 \
  --cat_out $OUT || { echo "scoring failed"; exit 4; }
# identity check (PROTOCOL_CAT.md); the 4-round trial's renders are the floor for the full run
# (Lab4D warms up over 2 rounds, so a run needs at least 3; see the protocol's Changes)
TRIAL=$D/experiments_cat/fg-skel-heldout-r4
TARG=""; [ "$ROUNDS" != 4 ] && [ -f $TRIAL/frames.npz ] && TARG="--trial $TRIAL"
PETS_ROOT=$D python $D/tools/cat_identity.py $OUT $TARG || echo "identity check failed (scores above still stand)"
# turntable at the used frame in the middle of the first used block (raw id of filtered frame A/2)
F=$(ls database/processed/JPEGImages/Full-Resolution/$SEQ-0000/*.jpg | sed -n "$(( A / 2 + 1 ))p" | xargs basename | sed 's/.jpg//; s/^0*//')
python lab4d/render.py --flagfile=logdir/$SEQ-$LOG/opts.log --load_suffix latest --viewpoint rot-0-360 \
  --render_res 256 --freeze_id ${F:-0} --num_frames 48 || { echo "turntable failed"; exit 5; }
cp logdir/$SEQ-$LOG/renderings_0000/rot-0-360/rgb.mp4 $OUT/turntable.mp4 2>/dev/null
cp logdir/$SEQ-$LOG/opts.log $OUT/
echo "DONE cat $LOG $(date)"
