#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH --gpus-per-node=1
#SBATCH -J claude-pika18x
# The rebuild from this cat's close-up clips (PROTOCOL_CAT.md): export the trained model (pika18_export.py), then the
# checks on CPU: R1/R2 (pika18_check.py), the videos' pair measures, R3/R4 (judge_recon2.py, pika18_score.py).
#   sbatch -A ACCOUNT -p a10 -q normal --time=01:00:00 -o logs/pika18x-%j.out run_pika18_export.sbatch ROUNDS
ROUNDS=${1:-160}
D=${PETS_ROOT:-/scratch/CLUSTER/$USER/claude-pets}
type module >/dev/null 2>&1 || source /etc/profile
module load gcc/11.5.0 cuda/12.1.1 || { echo 'module load failed'; exit 1; }
set -u
export PATH=$D/env-lab4d/bin:$PATH PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
unset PYTHONHOME PYTHONPATH
export TORCH_HOME=$D/cache/torch MPLCONFIGDIR=$D/cache/mpl DECIMATE_PY=$D/env-a2s/bin/python OPENBLAS_NUM_THREADS=1
LOG=fg-skel-r$ROUNDS; OUT=$D/experiments_pika18/$LOG
mkdir -p $OUT
cd $D/code/lab4d || exit 2
python $D/tools/pika18_export.py --flagfile=logdir/pika18-$LOG/opts.log --load_suffix latest --cat_out $OUT/export \
  || { echo "export failed"; exit 3; }
cd $D || exit 2
python tools/pika18_check.py tools $OUT/export code/lab4d/database/processed $OUT/check || { echo "check failed"; exit 4; }
J=experiments_judge
BORDER_OK=1 ONLY=cat-85-0066,cat-85-0071,cat-85-0072,cat-85-0073,cat-85-0055,cat-85-0060 python tools/judge_real.py $J/real $OUT/real_walk.npz \
  || { echo "video measures failed"; exit 5; }
ITEMS=""
for c in 0066 0071 0072 0073 0055 0060; do
  python tools/judge_recon2.py $OUT/real_walk.npz $OUT/real_walk.json cat-85-$c $OUT/export/cam_cat-85-$c.npz $OUT/recon2_$c.npz \
    && ITEMS="$ITEMS $c:cat-85-$c:$OUT/recon2_$c.npz"
done
python tools/judge_recon2.py $J/real_t2_pika.npz $J/real_t2_pika.json cat-pikachu-0-0000 $OUT/export/cam_cat-pikachu-0-0000.npz $OUT/recon2_body.npz \
  && ITEMS="$ITEMS body:cat-pikachu-0-0000:$OUT/recon2_body.npz"
python tools/pika18_score.py $J/real_pairs.npz $J/ours/walk_shown.npz+$J/ours/rec_shown.npz $OUT/r3_r4.json $ITEMS
echo "DONE pika18x $LOG $(date)"
