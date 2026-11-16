#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 6
#SBATCH --mem=48G
#SBATCH --gpus-per-node=1
#SBATCH -J claude-b3
#SBATCH -o logs/b3-%j.out
# Step B3 (PROTOCOL_BEHAVIOUR.md, "B3"): walks from training moments at 11 place chances, pick the one whose movement
# matches the real cat's in the training videos, score it on B2's held-back windows (context), then remake the page's
# 16 free walks with it and redo body motion for every walk. usage (from the pets folder):
#   sbatch -A ACCOUNT -p a10 -q normal --time=01:30:00 run_b3.sbatch LOG      (LOG = b128)
LOG=${1:-b128}
D=${PETS_ROOT:-/scratch/CLUSTER/$USER/claude-pets}
type module >/dev/null 2>&1 || source /etc/profile
module load gcc/11.5.0 cuda/12.1.1 || { echo 'module load failed'; exit 1; }
set -u
A=$D/code/agent2sim
NAME=cat-pikachu-2024-08-v2-compose-ft
export PATH=$D/env-a2s/bin:$PATH PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
unset PYTHONHOME
export PYTHONPATH=$A/lab4d PYOPENGL_PLATFORM=egl DECIMATE_PY=$D/env-a2s/bin/python
export TORCH_HOME=$D/cache/torch MPLCONFIGDIR=$D/cache/mpl
OUT=$D/experiments_b3/$LOG
mkdir -p $OUT
HELD=$D/data/a2s/motion/$NAME-heldout-L64-S10.pkl
TRAIN=$D/data/a2s/motion/$NAME-train-L64-S1.pkl
SRC=$D/dl/a2s/$NAME-train-L64-S1.pkl
PRIOR=$D/experiments_b2/$LOG/places_prior.json
ARGS="--load_logname $NAME --logname_gd $LOG --suffix latest"
WS="0 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5"
cd $A/gdmdm || exit 2
echo "B3 $LOG $(hostname) $(date)"
for W in $WS; do
  B1_OUT=$OUT B1_NAME=tune_w$W B1_STARTS=$TRAIN B2_PRIOR=$PRIOR B2_W=$W \
    python $D/tools/b1_rollout.py $ARGS --eval_batch_size 1 > $OUT/tune_w$W.log 2>&1 || { echo "walks at w=$W failed"; exit 3; }
  echo "walks at w=$W done $(date +%T)"
done
WP=$(python $D/tools/b3_pick.py $OUT $WS) || { echo "pick failed"; exit 4; }
echo "picked w = $WP"
B1_HELDOUT=$HELD B2_OUT=$OUT/score B2_EXTRA_W="$WP" python $D/tools/b2_goal.py $ARGS --eval_batch_size 192 --eval_ntrial 12 \
  > $OUT/score.log 2>&1 || { echo "scoring failed"; exit 5; }
grep -E "^method|^copy_w|^location|^replay" $OUT/score.log
B1_OUT=$OUT B1_NAME=walks_free_b3 B1_STARTS=$HELD B2_PRIOR=$PRIOR B2_W=$WP \
  python $D/tools/b1_rollout.py $ARGS --eval_batch_size 1 || { echo "page walks failed"; exit 6; }
python - "$OUT" "$D/experiments_s2/$LOG/walks_all.json" <<'PY' || { echo "merge failed"; exit 7; }
import json, os, sys
out, old = sys.argv[1], sys.argv[2]
free = json.load(open(os.path.join(out, "walks_free_b3.json")))["runs"]
runs = free + [r for r in json.load(open(old))["runs"] if r.get("kind", "free") != "free"]
json.dump({"fps": 10, "runs": runs}, open(os.path.join(out, "walks_all.json"), "w"))
print("walks:", {k: sum(r["kind"] == k for r in runs) for k in ("free", "spot", "replay", "call")})
src = [s for r in free for s in r["goal_src"]]
print("page free-walk rounds: %d, goal from a place in %d" % (len(src), src.count("place")))
PY
python $D/tools/a2s_cat_export.py $NAME $SRC $OUT 30000 $OUT/walks_all.json || { echo "body motion failed"; exit 8; }
echo "DONE b3 $LOG $(date)"
