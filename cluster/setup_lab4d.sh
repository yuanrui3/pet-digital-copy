#!/bin/bash
# Builds the Lab4D environment for the cat test (work/pets/PROTOCOL_CAT.md). Run once, from the pets folder:
#   cd /scratch/CLUSTER/$USER/claude-pets && bash setup_lab4d.sh > logs/setup_lab4d.log 2>&1
# Ends with LAB4D_ENV_DONE when everything imports; the EXIT line shows the exit code either way.
D=${PETS_ROOT:-/scratch/CLUSTER/$USER/claude-pets}
trap 'echo "LAB4D_ENV_EXIT $? $(date)"' EXIT
type module >/dev/null 2>&1 || source /etc/profile  # a non-interactive SSH shell has no module command
module load conda/2026.03 2>/dev/null || module load conda || { echo "no conda module"; exit 1; }
module load gcc/11.5.0 cuda/12.1.1 || { echo "no gcc/cuda module"; exit 1; }
export CUDA_HOME=$(dirname "$(dirname "$(which nvcc)")")
echo "nvcc: $(which nvcc)"
set -u
export CONDA_PKGS_DIRS=$D/cache/conda-pkgs PIP_CACHE_DIR=$D/cache/pip PYTHONNOUSERSITE=1
export TORCH_HOME=$D/cache/torch
E=$D/env-lab4d
[ -d $E ] || conda create -y -q -p $E python=3.10 pip || exit 2
export PATH=$E/bin:$PATH
unset PYTHONHOME PYTHONPATH
pip install -q torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121 || exit 3
pip install -q "numpy<2" absl-py trimesh scikit-image scipy opencv-python-headless einops numba tqdm \
  omegaconf matplotlib imageio imageio-ffmpeg tensorboard psutil lpips pybind11 timm==1.0.9 || exit 4
pip install -q --no-build-isolation pysdf || exit 5

# Lab4D at the commit named in the protocol, plus my patch (held-back colour; C++17 for the CUDA extension)
mkdir -p $D/code && cd $D/code
[ -d lab4d ] || git clone -q https://github.com/lab4d-org/lab4d.git || exit 6
cd lab4d && git checkout -q e62976600bcc4b6f4f08cf059701c1c603c9c34a || exit 7
git apply --check $D/tools/lab4d_heldout.patch 2>/dev/null && git apply $D/tools/lab4d_heldout.patch
git diff --stat
# the quaternion CUDA extension; A10 = compute capability 8.6 (no GPU on the login node, so name it)
(cd lab4d/third_party/quaternion && CUDA_HOME=$CUDA_HOME TORCH_CUDA_ARCH_LIST=8.6 pip install -q --no-build-isolation .) || exit 8
[ -e database ] || ln -s $D/data/lab4d/database database
# lab4d/utils/profile_utils.py imports a CPython test helper that only tidies tracebacks; fall back if absent
python -c "from _testcapi import set_exc_info" 2>/dev/null || sed -i \
  's/^from _testcapi import set_exc_info$/try:\n    from _testcapi import set_exc_info\nexcept ImportError:\n    set_exc_info = lambda *a: None/' \
  lab4d/utils/profile_utils.py

python - <<'EOF' || exit 9
import torch, _quaternion, pysdf, lpips, absl, trimesh, skimage, einops, omegaconf, timm
timm.create_model("swin_large_patch4_window12_384", pretrained=False, num_classes=0)  # the identity model's body
import lab4d.config, lab4d.engine.trainer, lab4d.engine.model
print("torch", torch.__version__, "cuda", torch.version.cuda)
lpips.LPIPS(net="vgg", verbose=False)
print("LAB4D_ENV_DONE")
EOF
