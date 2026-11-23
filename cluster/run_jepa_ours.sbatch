#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH --mem=96G
#SBATCH --gpus-per-node=1
#SBATCH -J claude-jepa-ours
# The video-model judge (PROTOCOL_CAT.md, "A video-model judge"): our silhouette clips for every version and the
# defect ladder, CPU only (the GPU is only how the queue hands out nodes), one process per set.
#   sbatch -A ACCOUNT -p a10 -q normal --time=00:40:00 -o logs/jepa-ours-%j.out run_jepa_ours.sbatch [NAME ...]
D=${PETS_ROOT:-/scratch/CLUSTER/$USER/claude-pets}
type module >/dev/null 2>&1 || source /etc/profile
set -u
export PATH=$D/env-lab4d/bin:$PATH PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
unset PYTHONHOME PYTHONPATH
export MPLCONFIGDIR=$D/cache/mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
cd $D || exit 2
O=experiments_jepa/clips
mkdir -p $O
A="experiments_feet/page experiments_feet/ik2 experiments_feet/floor experiments_head/g15/graft.npz experiments_judge/ours/walk_shown_windows.json"
G=experiments_jepa/dense
# name  source  defect
cat > $O/sets.txt <<EOF
v22 v22 none
g5 $G/g5/gait_dense.npz none
g9 $G/g9/gait_dense.npz none
g12 $G/g12/gait_dense.npz none
slide4 $G/slide4/gait_dense.npz none
slide8 $G/slide8/gait_dense.npz none
slide16 $G/slide16/gait_dense.npz none
stiff50 $G/g12/gait_dense.npz stiff:0.5
stiff100 $G/g12/gait_dense.npz stiff:1.0
hold6 $G/g12/gait_dense.npz hold:6
hold12 $G/g12/gait_dense.npz hold:12
jitter1 $G/g12/gait_dense.npz jitter:1
jitter3 $G/g12/gait_dense.npz jitter:3
jitter6 $G/g12/gait_dense.npz jitter:6
back $G/g12/gait_dense.npz back
EOF
want=" $* "
while read -r name src defect; do
  [ "$#" -gt 0 ] && [[ "$want" != *" $name "* ]] && continue
  ( SOURCE=$src DEFECT=$defect python tools/jepa_ours.py $A $O/$name.npz > $O/$name.log 2>&1; echo "$name exit $? $(date)" ) &
done < $O/sets.txt
wait
echo "DONE jepa ours $(date)"
