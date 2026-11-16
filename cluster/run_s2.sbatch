#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 6
#SBATCH --mem=48G
#SBATCH --gpus-per-node=1
#SBATCH -J claude-s2
#SBATCH -o logs/s2-%j.out
# Step S2 (PROTOCOL_STIMULUS.md): the room and a walking person as stimuli (check), the 12 places and the
# "call the cat" walks, then body motion for every walk (S1's walks plus these). usage (from the pets folder):
#   sbatch -A ACCOUNT -p a10 -q normal --time=01:30:00 run_s2.sbatch LOG      (LOG = b128)
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
OUT=$D/experiments_s2/$LOG
HELD=$D/data/a2s/motion/$NAME-heldout-L64-S10.pkl
SRC=$D/dl/a2s/$NAME-train-L64-S1.pkl
ARGS="--load_logname $NAME --logname_gd $LOG --suffix latest"
cd $A/gdmdm || exit 2
echo "S2 $LOG $(hostname) $(date)"
B1_HELDOUT=$HELD S2_OUT=$OUT python $D/tools/b1_room.py $ARGS --eval_batch_size 16 || { echo "check failed"; exit 3; }
B1_MODE=call B1_OUT=$OUT B1_NAME=walks_call B1_STARTS=$HELD S2_PLACES=$OUT/places.json B1_ROUNDS=3 B1_SEEDS=1 \
  python $D/tools/b1_rollout.py $ARGS --eval_batch_size 1 || { echo "call walks failed"; exit 4; }
python - "$OUT" "$D/experiments_s1/$LOG/walks_all.json" <<'PY' || { echo "merge failed"; exit 6; }
import json, os, sys
out, s1 = sys.argv[1], sys.argv[2]
runs = json.load(open(s1))["runs"] + json.load(open(os.path.join(out, "walks_call.json")))["runs"]
json.dump({"fps": 10, "runs": runs}, open(os.path.join(out, "walks_all.json"), "w"))
print("walks:", {k: sum(r["kind"] == k for r in runs) for k in ("free", "spot", "replay", "call")})
PY
python $D/tools/a2s_cat_export.py $NAME $SRC $OUT 30000 $OUT/walks_all.json || { echo "body motion failed"; exit 7; }
echo "DONE s2 $LOG $(date)"
