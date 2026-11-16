#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 6
#SBATCH --mem=48G
#SBATCH --gpus-per-node=1
#SBATCH -J claude-b2
#SBATCH -o logs/b2-%j.out
# Step B2 (PROTOCOL_BEHAVIOUR.md, "B2"): score "copy + places" goal choice on the held-back windows; if it passes,
# make the 16 free walks again with it and redo body motion for every walk on the page (S1's and S2's walks kept).
# usage (from the pets folder):  sbatch -A ACCOUNT -p a10 -q normal --time=01:00:00 run_b2.sbatch LOG   (LOG = b128)
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
OUT=$D/experiments_b2/$LOG
HELD=$D/data/a2s/motion/$NAME-heldout-L64-S10.pkl
SRC=$D/dl/a2s/$NAME-train-L64-S1.pkl
ARGS="--load_logname $NAME --logname_gd $LOG --suffix latest"
cd $A/gdmdm || exit 2
echo "B2 $LOG $(hostname) $(date)"
B1_HELDOUT=$HELD B2_OUT=$OUT python $D/tools/b2_goal.py $ARGS --eval_batch_size 192 --eval_ntrial 12 \
  || { echo "scoring failed"; exit 3; }
python -c "import json,sys; sys.exit(0 if json.load(open('$OUT/b2_results.json'))['passes'] else 1)" \
  || { echo "DONE b2 $LOG: does not pass, page walks unchanged $(date)"; exit 0; }
B1_OUT=$OUT B1_NAME=walks_free_b2 B1_STARTS=$HELD B2_PRIOR=$OUT/places_prior.json B2_W=0.5 \
  python $D/tools/b1_rollout.py $ARGS --eval_batch_size 1 || { echo "free walks failed"; exit 4; }
python - "$OUT" "$D/experiments_s2/$LOG/walks_all.json" <<'PY' || { echo "merge failed"; exit 5; }
import json, os, sys
out, old = sys.argv[1], sys.argv[2]
free = json.load(open(os.path.join(out, "walks_free_b2.json")))["runs"]
runs = free + [r for r in json.load(open(old))["runs"] if r.get("kind", "free") != "free"]
json.dump({"fps": 10, "runs": runs}, open(os.path.join(out, "walks_all.json"), "w"))
print("walks:", {k: sum(r["kind"] == k for r in runs) for k in ("free", "spot", "replay", "call")})
src = [s for r in free for s in r["goal_src"]]
print("free-walk rounds: %d, goal from a place in %d" % (len(src), src.count("place")))
PY
python $D/tools/a2s_cat_export.py $NAME $SRC $OUT 30000 $OUT/walks_all.json || { echo "body motion failed"; exit 6; }
echo "DONE b2 $LOG $(date)"
