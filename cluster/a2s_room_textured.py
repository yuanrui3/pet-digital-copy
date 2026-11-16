"""Exports the cat's home scan at full resolution with its photo texture, for the web viewer (demo, not a test).

usage (env-a2s, CPU): python a2s_room_textured.py POLYCAM_DIR OUT_DIR [TEXTURE_MAX_PX]
Writes OUT_DIR/room_tex.bin + room_tex.json (positions in the cat's world frame = scan * [1, -1, -1], as Agent-to-Sim
places it; texture coordinates; triangles) and OUT_DIR/room_tex.jpg.
"""
import json
import os
import sys

import numpy as np
import trimesh
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a2s_room_export import pack  # noqa: E402


def main(poly_dir, out_dir, tex_max=4096):
    os.makedirs(out_dir, exist_ok=True)
    ply = trimesh.load(os.path.join(poly_dir, "raw.ply"), process=False)
    print("ply:", len(ply.vertices), "vertices", len(ply.faces), "faces, visual", ply.visual.kind,
          "uv" if getattr(ply.visual, "uv", None) is not None else "no uv", flush=True)
    obj = trimesh.load(os.path.join(poly_dir, "raw.obj"), process=False, force="mesh")
    print("obj:", len(obj.vertices), "vertices", len(obj.faces), "faces, visual", obj.visual.kind,
          "uv" if getattr(obj.visual, "uv", None) is not None else "no uv", flush=True)
    print("bbox ply", ply.bounds.round(3).tolist(), "obj", obj.bounds.round(3).tolist(), flush=True)
    # the obj is NOT in the ply's frame (its bounds are symmetric, so they match even though it is upside down);
    # use the ply, which is the file Agent-to-Sim places, with its own per-vertex texture coordinates
    print("first vertices ply", np.asarray(ply.vertices[:2]).round(3).tolist(), "obj",
          np.asarray(obj.vertices[:2]).round(3).tolist(), flush=True)
    src = ply
    v = (np.asarray(src.vertices) * np.array([[1.0, -1.0, -1.0]])).astype(np.float32)
    uv = np.asarray(src.visual.uv, dtype=np.float32)
    f = np.asarray(src.faces, dtype=np.uint32)
    img = Image.open(os.path.join(poly_dir, "texture.png")).convert("RGB")
    s = min(1.0, tex_max / max(img.size))
    if s < 1:
        img = img.resize((round(img.size[0] * s), round(img.size[1] * s)), Image.LANCZOS)
    img.save(os.path.join(out_dir, "room_tex.jpg"), quality=86, optimize=True)
    meta, size = pack(os.path.join(out_dir, "room_tex.bin"), [("pos", v), ("uv", uv), ("faces", f)])
    with open(os.path.join(out_dir, "room_tex.json"), "w") as fh:
        json.dump({"blobs": meta, "bytes": size, "texture": "room_tex.jpg", "texture_size": list(img.size),
                   "frame": "cat world (Agent-to-Sim), y down", "bbox": [v.min(0).tolist(), v.max(0).tolist()]}, fh)
    print("room_tex: %d vertices, %d faces, %d bytes; texture %s, %d bytes" % (
        len(v), len(f), size, img.size, os.path.getsize(os.path.join(out_dir, "room_tex.jpg"))))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], *(int(x) for x in sys.argv[3:4]))
