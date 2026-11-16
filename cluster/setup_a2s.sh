#!/bin/bash
# Builds a separate environment for the behaviour test (work/pets/PROTOCOL_BEHAVIOUR.md): Agent-to-Sim's behaviour
# model plus the Lab4D branch it pins. Run once, from the pets folder:
#   cd /scratch/CLUSTER/$USER/claude-pets && bash setup_a2s.sh > logs/setup_a2s.log 2>&1
# Ends with A2S_ENV_DONE when everything imports; the A2S_ENV_EXIT line shows the exit code either way.
# The cat test's environment (env-lab4d) is not touched.
D=${PETS_ROOT:-/scratch/CLUSTER/$USER/claude-pets}
trap 'echo "A2S_ENV_EXIT $? $(date)"' EXIT
type module >/dev/null 2>&1 || source /etc/profile  # a non-interactive SSH shell has no module command
module load conda/2026.03 2>/dev/null || module load conda || { echo "no conda module"; exit 1; }
module load gcc/11.5.0 cuda/12.1.1 || { echo "no gcc/cuda module"; exit 1; }
export CUDA_HOME=$(dirname "$(dirname "$(which nvcc)")")
echo "nvcc: $(which nvcc)"
set -u
export CONDA_PKGS_DIRS=$D/cache/conda-pkgs PIP_CACHE_DIR=$D/cache/pip PYTHONNOUSERSITE=1
export TORCH_HOME=$D/cache/torch
E=$D/env-a2s
[ -d $E ] || conda create -y -q -p $E python=3.10 pip || exit 2
export PATH=$E/bin:$PATH
unset PYTHONHOME PYTHONPATH
pip install -q torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121 || exit 3
pip install -q "numpy<2" absl-py trimesh scikit-image scipy opencv-python-headless einops numba tqdm omegaconf \
  matplotlib imageio imageio-ffmpeg tensorboard psutil pybind11 open3d viser pyrender PyOpenGL recordclass celluloid \
  geomloss point-cloud-utils || exit 4
pip install -q --no-build-isolation pysdf || exit 5
# urdfpy reads the cat's skeleton file; Agent-to-Sim's README then pins networkx 2.5
pip install -q urdfpy && pip install -q "networkx==2.5" || exit 6
pip install -q git+https://github.com/patrick-kidger/torchcubicspline.git || exit 7

# Agent-to-Sim at the protocol's commit; its Lab4D submodule (lab4dgs branch) at the commit it pins
mkdir -p $D/code && cd $D/code
[ -d agent2sim/.git ] || git clone -q https://github.com/facebookresearch/agent2sim.git || exit 8
cd agent2sim && git checkout -q 1952ed13868ec4419e349c0298e8e2a4d49cbfd3 || exit 9
A=$D/code/agent2sim
[ -d lab4d/.git ] || { rmdir lab4d 2>/dev/null; git clone -q --branch lab4dgs https://github.com/lab4d-org/lab4d.git lab4d; } || exit 10
cd lab4d && git checkout -q 334c1b4866a181fdbec24f7cf4a6e006c4070b41 || exit 11
git submodule update --init -q projects/ppr/ppr-diffphys || exit 12
# this branch's quaternion CUDA extension (its source differs from the cat test's); C++17 for PyTorch 2.4
(cd lab4d/third_party/quaternion && sed -i 's/-std=c++14/-std=c++17/g' setup.py \
  && TORCH_CUDA_ARCH_LIST=8.6 pip install -q --no-build-isolation .) || exit 13
python -c "from _testcapi import set_exc_info" 2>/dev/null || sed -i \
  's/^from _testcapi import set_exc_info$/try:\n    from _testcapi import set_exc_info\nexcept ImportError:\n    set_exc_info = lambda *a: None/' \
  lab4d/utils/profile_utils.py
git -C $A/lab4d status --short | head

# the released cat reconstruction (room, observer) and its cached data summary, where the code looks for them
mkdir -p logdir tmp database/motion
[ -e logdir/cat-pikachu-2024-08-v2-compose-ft ] || ln -s $D/data/a2s/logdir/cat-pikachu-2024-08-v2-compose-ft logdir/
[ -e tmp/cat-pikachu-2024-08-v2-compose-ft.pkl ] || ln -s $D/data/a2s/tmp/cat-pikachu-2024-08-v2-compose-ft.pkl tmp/
cd $A/gdmdm
for k in logdir database tmp; do [ -e $k ] || ln -s ../lab4d/$k $k; done

# urdfpy predates NumPy 1.24; if loading the skeleton trips on a removed alias, map it to the 64-bit type
Q=$A/lab4d/projects/ppr/ppr-diffphys/data/urdf_templates/quad.urdf
if ! python -c "from urdfpy import URDF; URDF.load('$Q')" 2>/tmp/a2s_urdf.err; then
  tail -2 /tmp/a2s_urdf.err
  U=$(python -c "import urdfpy, os; print(os.path.dirname(urdfpy.__file__))")
  sed -i -E 's/np\.float([^0-9_a-zA-Z]|$)/np.float64\1/g; s/np\.int([^0-9_a-zA-Z]|$)/np.int64\1/g' $U/*.py
  python -c "from urdfpy import URDF; URDF.load('$Q')" || exit 14
  echo "urdfpy: replaced removed NumPy aliases"
fi

cd $A/gdmdm
PYTHONPATH=$A/lab4d PYOPENGL_PLATFORM=egl python - <<'EOF' || exit 15
import torch, _quaternion, open3d, viser, recordclass, celluloid, torchcubicspline, geomloss, point_cloud_utils
print("torch", torch.__version__, "cuda", torch.version.cuda)
try:
    import pyrender  # only drawing uses it; the GPU nodes have EGL, a login node may not
    print("pyrender ok")
except Exception as e:
    print("pyrender import failed here (login node?):", repr(e)[:200])
import config, ddpm, arch
from projects.csim.voxelize import BGField
print("A2S_ENV_DONE")
EOF
pip freeze 2>/dev/null | grep -i -E "^(torch|numpy|open3d|viser|pyrender|urdfpy|networkx|recordclass|torchcubicspline|geomloss)"
