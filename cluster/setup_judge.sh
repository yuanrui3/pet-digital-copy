#!/bin/bash
# Builds the environment for the video-model judge (work/pets/PROTOCOL_CAT.md, "A video-model judge"):
# SAM 2.1 outlines the real cat-85 frames, V-JEPA 2 turns silhouette clips into motion features.
# Both need newer software than the other environments have (torch >= 2.5.1, transformers >= 4.53).
# It installs third-party code, so a person starts it, not the agent. CPU only; run it on a login node:
#   cd /scratch/CLUSTER/$USER/claude-pets && bash setup_judge.sh > logs/setup_judge.log 2>&1
# The last line of the log says how it ended: JUDGE_ENV_DONE, then JUDGE_ENV_EXIT with the exit code.
# The other environments (env, env-lab4d, env-a2s) are not touched.
D=${PETS_ROOT:-/scratch/CLUSTER/$USER/claude-pets}
trap 'echo "JUDGE_ENV_EXIT $? $(date)"' EXIT
type module >/dev/null 2>&1 || source /etc/profile  # a non-interactive SSH shell has no module command
module load conda/2026.03 2>/dev/null || module load conda || { echo "no conda module"; exit 1; }
set -u
export CONDA_PKGS_DIRS=$D/cache/conda-pkgs PIP_CACHE_DIR=$D/cache/pip PYTHONNOUSERSITE=1
export TORCH_HOME=$D/cache/torch HF_HOME=$D/cache/hf HF_HUB_OFFLINE=1
mkdir -p $D/cache $D/code $D/logs
E=$D/env-judge
[ -x $E/bin/python ] || conda create -y -q -p $E python=3.11 pip || exit 2
export PATH=$E/bin:$PATH
unset PYTHONHOME PYTHONPATH
pip install -q torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121 || exit 3
# the last 4.x transformers (V-JEPA 2 arrived in 4.53); 5.x may want a newer torch
pip install -q "numpy<2" "transformers==4.57.6" safetensors scipy pillow opencv-python-headless \
  imageio imageio-ffmpeg tqdm || exit 4
# SAM 2 at a fixed commit, without its optional CUDA extension (it only fills small holes in masks)
cd $D/code
[ -d sam2/.git ] || git clone -q https://github.com/facebookresearch/sam2.git || exit 5
(cd sam2 && git checkout -q 2b90b9f5ceec907a1c18123530e92e794ad901a4 && git log -1 --format='sam2 %H %ad')
SAM2_BUILD_CUDA=0 pip install -q --no-build-isolation -e ./sam2 || exit 6
pip install -q "numpy<2"
cd $D
python - <<'PY' || exit 7
import torch, torchvision, transformers, numpy
print("torch", torch.__version__, "cuda", torch.version.cuda, "torchvision", torchvision.__version__,
      "transformers", transformers.__version__, "numpy", numpy.__version__)
from transformers import AutoModel, AutoVideoProcessor
from sam2.build_sam import build_sam2_video_predictor
print("JUDGE_ENV_DONE")
PY
pip freeze 2>/dev/null | grep -i -E "^(torch|torchvision|transformers|numpy|sam-2|sam2|opencv)"
