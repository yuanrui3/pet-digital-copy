#!/bin/bash
# Downloads the judge's two public models (approved by the user on 2026-11-22):
#   V-JEPA 2 ViT-L, facebook/vjepa2-vitl-fpc64-256 (MIT), model weights + configs only, ~1.30 GB
#   SAM 2.1 small, Meta's sam2.1_hiera_small.pt (Apache 2.0), ~184 MB
# Run with nohup on a login node; the log ends with DL_DONE or DL_FAIL.
D=${PETS_ROOT:-/scratch/CLUSTER/$USER/claude-pets}
M=$D/models/vjepa2-vitl-fpc64-256
S=$D/models/sam2
mkdir -p $M $S
R=https://huggingface.co/facebook/vjepa2-vitl-fpc64-256/resolve/b3c1679b7c34d3255ef3547f27c7b226aefab26f
for f in config.json video_preprocessor_config.json model.safetensors; do
  [ -s $M/$f ] || { curl -sSfL --retry 3 -o $M/$f.part $R/$f && mv -f $M/$f.part $M/$f; }
  [ -s $M/$f ] || { echo "DL_FAIL $f"; exit 1; }
done
[ -s $S/sam2.1_hiera_small.pt ] || { curl -sSfL --retry 3 -o $S/sam2.1_hiera_small.pt.part \
  https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt \
  && mv -f $S/sam2.1_hiera_small.pt.part $S/sam2.1_hiera_small.pt; }
[ -s $S/sam2.1_hiera_small.pt ] || { echo "DL_FAIL sam2"; exit 1; }
ls -l $M $S
sha256sum $M/model.safetensors $S/sam2.1_hiera_small.pt
echo DL_DONE
