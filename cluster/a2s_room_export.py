"""Exports the cat's home (Agent-to-Sim's phone scan) and the cat's recorded paths for the web viewer (demo, not a test).

usage (env-a2s, CPU): python a2s_room_export.py MOTION_PKL POLYCAM_PLY OUT_DIR [ROOM_FACES]
Writes OUT_DIR/room.bin + room.json (decimated, coloured room without its ceiling, in the cat's world frame) and
OUT_DIR/paths.bin + paths.json (per video: the cat's root position and heading and the observer's position, 10 fps;
the cat's skeleton joints if present).

The scan is placed in the cat's world frame exactly as Agent-to-Sim does (vertices * [1, -1, -1], BGField with
use_default_mesh=True). Videos are recovered from the released file as in b1_split.py.
"""
import json
import os
import pickle
import sys

import numpy as np
import open3d as o3d
import trimesh

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from b1_split import KEYS, same_video  # noqa: E402


def pack(path, blobs):
    meta, off = [], 0
    with open(path, "wb") as fh:
        for name, arr in blobs:
            b = np.ascontiguousarray(arr).tobytes()
            meta.append({"name": name, "dtype": str(arr.dtype), "shape": list(arr.shape), "offset": off,
                         "bytes": len(b)})
            fh.write(b + b"\0" * ((-len(b)) % 4))
            off += len(b) + (-len(b)) % 4
    return meta, off


def main(pkl_path, ply_path, out_dir, room_faces=80000):
    os.makedirs(out_dir, exist_ok=True)
    # ---- room
    scan = trimesh.load(ply_path, process=False)
    print("scan:", type(scan).__name__, len(scan.vertices), "vertices", len(scan.faces), "faces;",
          "vertex colours" if scan.visual.kind == "vertex" else "visual kind %s" % scan.visual.kind, flush=True)
    v = np.asarray(scan.vertices) * np.array([[1.0, -1.0, -1.0]])
    col = np.asarray(scan.visual.vertex_colors)[:, :3] if scan.visual.kind == "vertex" else \
        np.asarray(scan.visual.to_color().vertex_colors)[:, :3]
    m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v), o3d.utility.Vector3iVector(np.asarray(scan.faces)))
    m.vertex_colors = o3d.utility.Vector3dVector(col / 255.0)
    m = m.simplify_quadric_decimation(int(room_faces))
    m.remove_unreferenced_vertices()
    m.remove_degenerate_triangles()
    rv, rf = np.asarray(m.vertices), np.asarray(m.triangles)
    rc = (np.clip(np.asarray(m.vertex_colors), 0, 1) * 255).round().astype(np.uint8)
    print("room after decimation:", len(rv), "vertices", len(rf), "faces", flush=True)

    # ---- paths
    with open(pkl_path, "rb") as f:
        d = pickle.load(f)
    n = len(d["se3"])
    for k in d:
        x = np.asarray(d[k][0])
        print("key %s: %d windows, window shape %s" % (k, len(d[k]), x.shape), flush=True)
    starts = [0] + [i + 1 for i in range(n - 1) if not same_video(d, i)]
    videos = list(zip(starts, starts[1:] + [n]))
    paths, root_all, cam_all, joints_all, heading_all = [], [], [], [], []
    for vi, (s, e) in enumerate(videos):
        def full(key):
            w = [np.asarray(d[key][s])] + [np.asarray(d[key][i])[-1:] for i in range(s + 1, e)]
            return np.concatenate(w, 0)
        world_to_root = full("se3")  # (T,4,4)
        root = np.linalg.inv(world_to_root)
        cam = np.linalg.inv(full("cam_se3"))
        T = len(root)
        root_all.append(root[:, :3, 3])
        cam_all.append(cam[:, :3, 3])
        heading_all.append(root[:, :3, :3].reshape(T, 9))
        if "joints3D" in d:
            joints_all.append(full("joints3D").reshape(T, -1, 3))
        paths.append({"video": vi, "frames": T, "seconds": T / 10.0})
    root_all = np.concatenate(root_all).astype(np.float32)
    cam_all = np.concatenate(cam_all).astype(np.float32)
    heading_all = np.concatenate(heading_all).astype(np.float32)

    # floor and up direction from the cat's own path: the cat walks on the floor
    up_guess = -np.array([0.0, 1.0, 0.0])  # y points down in this frame
    floor_h = float(np.percentile(root_all @ up_guess, 5))
    blobs_room = [("pos", rv.astype(np.float32)), ("col", rc), ("faces", rf.astype(np.uint32))]
    meta_room, size_room = pack(os.path.join(out_dir, "room.bin"), blobs_room)
    blobs_paths = [("root", root_all), ("cam", cam_all), ("heading", heading_all)]
    if joints_all:
        blobs_paths.append(("joints", np.concatenate(joints_all).astype(np.float16)))
    meta_paths, size_paths = pack(os.path.join(out_dir, "paths.bin"), blobs_paths)
    with open(os.path.join(out_dir, "room.json"), "w") as fh:
        json.dump({"blobs": meta_room, "bytes": size_room, "frame": "cat world (Agent-to-Sim), y down",
                   "bbox": [rv.min(0).tolist(), rv.max(0).tolist()]}, fh)
    with open(os.path.join(out_dir, "paths.json"), "w") as fh:
        json.dump({"blobs": meta_paths, "bytes": size_paths, "fps": 10, "videos": paths,
                   "root_height_5th_percentile_up": floor_h,
                   "root_bbox": [root_all.min(0).tolist(), root_all.max(0).tolist()],
                   "cam_bbox": [cam_all.min(0).tolist(), cam_all.max(0).tolist()]}, fh)
    print("room %d bytes, paths %d bytes, %d videos, %d frames" % (size_room, size_paths, len(paths), len(root_all)))
    print("room bbox", rv.min(0).round(2), rv.max(0).round(2))
    print("cat path bbox", root_all.min(0).round(2), root_all.max(0).round(2))


if __name__ == "__main__":
    main(*sys.argv[1:4], *(int(x) for x in sys.argv[4:5]))
