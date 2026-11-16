#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 6
#SBATCH --mem=48G
#SBATCH --gpus-per-node=1
#SBATCH -J claude-s1
#SBATCH -o logs/s1-%j.out
# Step S1, the person as a stimulus (PROTOCOL_STIMULUS.md): the check, the 12 spots and their walks, the replays
# of the held-back recordings, and body motion for every walk. usage (from the pets folder):
#   sbatch -A ACCOUNT -p a10 -q normal --time=02:00:00 run_s1.sbatch LOG [quick]      (LOG = b128)
LOG=${1:-b128}; QUICK=${2:-}
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
OUT=$D/experiments_s1/$LOG$QUICK
HELD=$D/data/a2s/motion/$NAME-heldout-L64-S10.pkl
SRC=$D/dl/a2s/$NAME-train-L64-S1.pkl
ARGS="--load_logname $NAME --logname_gd $LOG --suffix latest"
ROUNDS=6; SEEDS=2; RSEEDS=4; [ -n "$QUICK" ] && { ROUNDS=2; SEEDS=1; RSEEDS=1; }
cd $A/gdmdm || exit 2
echo "S1 $LOG $QUICK $(hostname) $(date)"
B1_HELDOUT=$HELD S1_OUT=$OUT python $D/tools/b1_stimulus.py $ARGS --eval_batch_size 16 || { echo "check failed"; exit 3; }
B1_MODE=spots B1_OUT=$OUT B1_NAME=walks_spots B1_STARTS=$HELD S1_SPOTS=$OUT/spots.json B1_ROUNDS=$ROUNDS B1_SEEDS=$SEEDS \
  python $D/tools/b1_rollout.py $ARGS --eval_batch_size 1 || { echo "spot walks failed"; exit 4; }
B1_MODE=replay B1_OUT=$OUT B1_NAME=walks_replay B1_SOURCE=$SRC B1_SEEDS=$RSEEDS \
  python $D/tools/b1_rollout.py $ARGS --eval_batch_size 1 || { echo "replay walks failed"; exit 5; }
# one file with every walk (tonight's free walks first, when they exist), then body motion for all of them
python - "$OUT" "$D/experiments_b1/$LOG/rollouts.json" <<'PY' || { echo "merge failed"; exit 6; }
import json, os, sys
out, free = sys.argv[1], sys.argv[2]
runs = json.load(open(free))["runs"] if os.path.exists(free) else []
runs = [r for r in runs if "joints" in r]  # walks made before joint angles were saved cannot be animated
for r in runs:
    r.setdefault("kind", "free")
for name in ("walks_spots", "walks_replay"):
    runs += json.load(open(os.path.join(out, name + ".json")))["runs"]
json.dump({"fps": 10, "runs": runs}, open(os.path.join(out, "walks_all.json"), "w"))
print("walks:", {k: sum(r["kind"] == k for r in runs) for k in ("free", "spot", "replay")})
PY
python $D/tools/a2s_cat_export.py $NAME $SRC $OUT 30000 $OUT/walks_all.json || { echo "body motion failed"; exit 7; }
echo "DONE s1 $LOG $QUICK $(date)"
