"""The whole Body tab cat at home, one-day test (work/pets/PROTOCOL_CAT.md): the Body tab cat's photo coat with the
same fill as the version 22 head (head_graft.py COVER): spots its video never saw, or much darker than the fur within
5 cm (outside the front 3 cm of the face), take a smooth fill from the reliable fur around them. Nothing else changes.
CPU only.
usage: python cat3_coat.py TOOLS_DIR CAT3_DIR C1_TEX C1_COVER OUT_JPG
  CAT3_DIR: a2s_cat3.bin/.json (home2_retarget.py NAME=a2s_cat3: its rest mesh in Pikachu's frame, metres)"""
import os
import sys
sys.path.insert(0, sys.argv[1])
import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates, uniform_filter
from home_coat import DARK_FROM, DARK_FULL, lightness, load_page, voxel_fill
from home2_coat import rasterise

cat3, c1_tex, c1_cover, out = sys.argv[2:6]
m, A = load_page(cat3, "a2s_cat3")
X, F, UV = A["pos"].astype(np.float64), A["faces"].astype(np.int64), A["uv"].astype(np.float64)
Tc = np.asarray(Image.open(c1_tex).convert("RGB"), dtype=np.float32) / 255.0
cov = np.asarray(Image.open(c1_cover).convert("L"), dtype=np.float32) / 255.0
S = Tc.shape[0]
tpos, trc = rasterise(X, F, UV, S)
photo = Tc[trc[:, 0], trc[:, 1]].copy()
seen = np.clip((cov[trc[:, 0], trc[:, 1]] - 0.02) / 0.08, 0, 1)
Lp = lightness(photo)
lo1 = tpos.min(0) - 0.03
ij1 = np.floor((tpos - lo1) / 0.01).astype(int)
accL, cntL = np.zeros(tuple(ij1.max(0) + 4)), np.zeros(tuple(ij1.max(0) + 4))
np.add.at(accL, tuple(ij1.T), seen * Lp)
np.add.at(cntL, tuple(ij1.T), seen)
accL, cntL = uniform_filter(accL, 5), uniform_filter(cntL, 5)
loc = map_coordinates(np.where(cntL > 1e-9, accL / np.maximum(cntL, 1e-9), 0), ((tpos - lo1) / 0.01 - 0.5).T, order=1, mode="nearest")
dark = np.clip((loc - Lp - DARK_FROM) / (DARK_FULL - DARK_FROM), 0, 1)
face_front = tpos[:, 2] - X[:, 2].min() < 0.03  # the cat faces -z in Pikachu's frame
dark[face_front] = 0
rel = seen * (1 - dark)
new = rel[:, None] * photo + (1 - rel[:, None]) * voxel_fill(photo, tpos, rel)
Tn = Tc.copy()
Tn[trc[:, 0], trc[:, 1]] = new
Image.fromarray((Tn * 255).round().clip(0, 255).astype(np.uint8)).save(out, quality=90)
print("coat fill: %d texels, %.0f%% unseen by its video, %.0f%% much darker than the fur round them; lightness under 35 "
      "outside the face %.1f%% -> %.1f%%" % (len(tpos), 100 * (seen < 0.5).mean(), 100 * (dark > 0.5).mean(),
                                               100 * ((Lp < 35) & ~face_front).mean(), 100 * ((lightness(new) < 35) & ~face_front).mean()))
