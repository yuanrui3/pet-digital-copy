"""The whole Body tab cat at home, one-day test, check 2 (work/pets/PROTOCOL_CAT.md): a contact sheet of 40 frames
spread over the recordings and the walks, plain grey with flat shading (shape only), the Body tab cat (C1) moved by
Pikachu's motion through the region retarget, as the page's poseGrey does it (planted-paw bending not applied here),
beside Pikachu's version 22 shape (home mesh with the grafted head). CPU only.
usage: python cat3_sheet.py TOOLS_DIR FEET_PAGE_DIR GRAFT_NPZ OUT_PNG
  FEET_PAGE_DIR: feet_check.py's page folder with a2s_cat2 = the C1 cat (a2s_cat3); GRAFT_NPZ: head_graft.py's graft.npz"""
import sys
sys.path.insert(0, sys.argv[1])
import numpy as np
from PIL import Image
from feet_check import Page
from home2_retarget import region_retarget
from home2_transfer import lbs
from home_coat import render

page_dir, graft, out = sys.argv[2:5]
pg = Page(page_dir)
g = np.load(graft)
grey = np.full((8, 8, 3), 0.8, np.float32)
uvq = np.zeros((len(pg.Q), 2))
Ps, H, Fh = g["Ps"].astype(np.float64), g["H"].astype(np.float64), g["Fh"].astype(np.int64)
hidx, hw = g["hidx"].astype(np.int64), g["hw"].astype(np.float64)
recs = [t for t in pg.tracks if t["kind"] == "rec"]
walks = [t for t in pg.tracks if t["kind"] == "walk" and t.get("sub") == "free"]
picks = []
for group in (recs, walks):
    tot = sum(len(t["bones"]) for t in group)
    for q in np.linspace(0.02, 0.98, 20):
        k, a = 0, int(q * tot)
        while a >= len(group[k]["bones"]):
            a -= len(group[k]["bones"])
            k += 1
        picks.append((group[k], a))
tiles = []
for tr, a in picks:
    T = tr["bones"][a:a + 1].astype(np.float64)
    Xc = lbs(pg.Q, pg.ig, pg.wg, region_retarget(T, pg.mix, pg.C)[0])
    Xp = np.concatenate([lbs(Ps, pg.ip, pg.wp, T[0]), lbs(H, hidx, hw, T[0])])
    Fp = np.concatenate([pg.fp, Fh + len(Ps)])
    row = [render(Xp, Fp, np.zeros((len(Xp), 2)), grey, "side", size=260), render(Xc, pg.fg, uvq, grey, "side", size=260)]
    tile = np.concatenate(row, 1)
    tile[:3], tile[:, :2] = 0.5, 0.5
    tiles.append(tile)
rows = [np.concatenate(tiles[i:i + 4], 1) for i in range(0, 40, 4)]
Image.fromarray((np.concatenate(rows, 0) * 255).clip(0, 255).astype(np.uint8)).save(out)
print("sheet: 40 frames (20 recorded, 20 walks); each tile: version 22 | the C1 cat", flush=True)
