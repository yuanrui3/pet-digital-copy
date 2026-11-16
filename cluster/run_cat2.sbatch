#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 6
#SBATCH --mem=48G
#SBATCH --gpus-per-node=1
#SBATCH -J claude-cat2
#SBATCH -o logs/cat2-%j.out
# A second cat for the viewer (demo, not a test): Lab4D on one video of Lab4D's preprocessed cat-85 set, all frames
# used, then the viewer export with baked and photo textures. usage (from the pets folder):
#   sbatch -A ACCOUNT -p a10 -q normal --time=04:00:00 run_cat2.sbatch SEQ ROUNDS STRIDE   (cat85-0068 60 2)
SEQ=${1:-cat85-0068}; ROUNDS=${2:-60}; STRIDE=${3:-2}
D=${PETS_ROOT:-/scratch/CLUSTER/$USER/claude-pets}
type module >/dev/null 2>&1 || source /etc/profile
module load gcc/11.5.0 cuda/12.1.1 || { echo 'module load failed'; exit 1; }
set -u
export PATH=$D/env-lab4d/bin:$PATH PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
unset PYTHONHOME PYTHONPATH
export TORCH_HOME=$D/cache/torch MPLCONFIGDIR=$D/cache/mpl DECIMATE_PY=$D/env-a2s/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $D/code/lab4d || exit 2
LOG=fg-skel-r$ROUNDS
OUT=$D/experiments_cat2/$SEQ-$LOG
mkdir -p $OUT
echo "cat2 $SEQ rounds $ROUNDS $(hostname) $(date)"
SAVE=10; [ $((ROUNDS % 10)) -eq 0 ] || SAVE=1
torchrun --nproc_per_node 1 --nnodes 1 --rdzv_backend c10d --rdzv_endpoint localhost:0 lab4d/train.py --ngpu 1 \
  --seqname $SEQ --logname $LOG --fg_motion skel-quad --num_rounds $ROUNDS --save_freq $SAVE \
  || { echo "training failed"; exit 3; }
echo "trained $(date)"
FLAGS="--flagfile=logdir/$SEQ-$LOG/opts.log --load_suffix latest"
# 1) rest mesh (grid 256, 44k faces), 2) UV atlas, 3) export with baked textures, 4) photo texture
python $D/tools/cat_viewer_export.py $FLAGS --cat_out $OUT/viewer --viewer_grid 256 --viewer_faces 44000 \
  --viewer_stride $STRIDE || { echo "export 1 failed"; exit 4; }
$DECIMATE_PY $D/tools/cat_uv_bake.py $OUT/viewer/rest.obj 2048 $OUT/uv.npz || { echo "uv bake failed"; exit 5; }
COLF=$(python -c "import json; m=json.load(open('$OUT/viewer/viewer_meta.json')); u=m['used_raw_ids']; print(u[len(u)//2])")
UV_NPZ=$OUT/uv.npz python $D/tools/cat_viewer_export.py $FLAGS --cat_out $OUT/viewer --viewer_grid 256 \
  --viewer_faces 44000 --viewer_stride $STRIDE --viewer_colour_frame $COLF || { echo "export 2 failed"; exit 6; }
UV_NPZ=$OUT/uv.npz NET_TEX=$OUT/viewer/cat_tex_fixed.jpg python $D/tools/cat_photo_texture.py $FLAGS \
  --cat_out $OUT/photo || { echo "photo texture failed"; exit 7; }
python lab4d/render.py $FLAGS --viewpoint rot-0-360 --render_res 256 --freeze_id $COLF --num_frames 48 \
  && cp logdir/$SEQ-$LOG/renderings_0000/rot-0-360/rgb.mp4 $OUT/turntable.mp4
echo "DONE cat2 $SEQ $LOG $(date)"
