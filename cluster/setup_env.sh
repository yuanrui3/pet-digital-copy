#!/bin/bash
# Builds the Python environment for 4D-Animal in its own cluster folder. CPU only; runs on the login node.
# It installs third-party code (PyTorch, PyTorch3D, Lightplane, 4D-Animal), so a person starts it, not the agent.
D=${PETS_ROOT:-/scratch/CLUSTER/$USER/claude-pets}
type module >/dev/null 2>&1 || source /etc/profile
module load conda/2026.03 || module load conda || { echo "no conda module"; exit 1; }
set -u
# the last line of the log always says how it ended: ENV_DONE, or ENV_EXIT with the failing step's code
trap 'echo "ENV_EXIT $? $(date)"' EXIT
export CONDA_PKGS_DIRS=$D/cache/conda-pkgs PIP_CACHE_DIR=$D/cache/pip PYTHONNOUSERSITE=1
export TORCH_HOME=$D/cache/torch
mkdir -p $D/cache $D/code $D/logs
[ -x $D/env/bin/python ] || conda create -y -q -p $D/env python=3.10 pip || { echo "conda create failed"; exit 1; }
PY=$D/env/bin/python
$PY -m pip install -q torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121 || exit 2
$PY -m pip install -q "numpy<2" || exit 2
# prebuilt PyTorch3D wheel for Python 3.10, CUDA 12.1, PyTorch 2.4.1; it is installed with --no-index,
# so its dependency iopath has to come from PyPI first
$PY -m pip install -q iopath || exit 3
$PY -m pip install -q --no-index --no-cache-dir pytorch3d \
  -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu121_pyt241/download.html || exit 3
$PY -m pip install -q cogapp plotly pandas sqlalchemy hydra-core tensorboard lpips opencv-python-headless \
  "imageio[ffmpeg]" pyyaml Pillow natsort tqdm || exit 4
cd $D/code
[ -d lightplane ] || git clone -q https://github.com/facebookresearch/lightplane || exit 5
(cd lightplane && git log -1 --format='lightplane %H %ad')
$PY -m pip install -q -e ./lightplane || exit 6
[ -d 4D-Animal ] || git clone -q https://github.com/zhongshsh/4D-Animal || exit 7
(cd 4D-Animal && git checkout -q 2b8a95968c1a17287697b6c1c1991bb1415a2097 && git log -1 --format='4D-Animal %H %ad')
$PY -m pip install -q "numpy<2"
# check the imports, and fetch the two public vision models the code downloads on first use
$PY - <<'PY' || exit 8
import torch, torchvision, pytorch3d, lightplane, lpips, numpy, triton
print("torch", torch.__version__, "torchvision", torchvision.__version__, "pytorch3d", pytorch3d.__version__,
      "numpy", numpy.__version__, "triton", triton.__version__)
from pytorch3d.implicitron.dataset.sql_dataset import SqlIndexDataset
print("implicitron sql dataset ok")
torch.hub.load("facebookresearch/dino:main", "dino_vits8")
lpips.LPIPS(net="vgg", verbose=False)
print("dino and lpips weights cached")
PY
echo "ENV_DONE $(date)"
