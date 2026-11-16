#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 6
#SBATCH --mem=32G
#SBATCH --gpus-per-node=1
#SBATCH -J claude-pets
#SBATCH -o logs/run-%j.out
# One reconstruction and its scoring.
# usage (from the pets folder): sbatch -A ACCOUNT -p a10 -q normal --time=04:00:00 run_scene.sbatch SEQ ARM [STEPS]
#   ARM A = 15 frames in, 5 held back, repeated (the authors' protocol); ARM B = frames 75-124 held back as a block
#   STEPS other than 10000 makes a smoke run in experiments_smoke/
SEQ=$1; ARM=$2; STEPS=${3:-10000}
D=${PETS_ROOT:-/scratch/CLUSTER/$USER/claude-pets}
type module >/dev/null 2>&1 || source /etc/profile
module load gcc/11.5.0 cuda/12.1.1 || { echo 'module load failed'; exit 1; }
set -u
export PATH=$D/env/bin:$PATH PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
unset PYTHONHOME PYTHONPATH
export TORCH_HOME=$D/cache/torch TRITON_CACHE_DIR=$D/cache/triton MPLCONFIGDIR=$D/cache/mpl
# on a 22 GB A10 the final scoring of 50 held-back frames ran out of memory through fragmentation (job 12377506)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p $TRITON_CACHE_DIR $MPLCONFIGDIR
case $ARM in A) TR=15; TE=5;; B) TR=75; TE=50;; *) echo "ARM must be A or B"; exit 2;; esac
OUT=$D/experiments/$ARM; EXTRA=""
if [ "$STEPS" != 10000 ]; then OUT=$D/experiments_smoke/$ARM; EXTRA="exp.n_steps=$STEPS exp.checkpoint_freq=$STEPS"; fi
cd $D/code/4D-Animal
[ -e external_data ] || ln -s $D/data/external_data external_data
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "START $SEQ arm $ARM steps $STEPS $(date)"
python main_optimize_scene.py "exp.sequence_index=\"$SEQ\"" "exp.experiment_folder=\"$OUT\"" \
  exp.train_split=$TR exp.test_split=$TE $EXTRA || { echo "optimise failed"; exit 3; }
echo "OPTIMISED $(date)"
# pet_eval.py lives in tools/: the pets folder itself has a data/ folder that would shadow the code's data package
python $D/tools/pet_eval.py score $OUT/$SEQ || { echo "score failed"; exit 4; }
python $D/tools/pet_eval.py orbit $OUT/$SEQ || { echo "orbit failed"; exit 5; }
echo "DONE $SEQ arm $ARM $(date)"
