"""UV-unwraps a cat mesh and finds the 3D surface point under every texel, so a colour field can be baked into a
texture (demo, not a test). Needs open3d (env-a2s).

uv_bake(mesh_path, size) -> dict with the welded mesh (pos (V,3), faces (F,3), uv (V,2); texture coordinates follow
three.js / OpenGL: v = 1 is the top row of the image) and the texels to fill: texel_pos (N,3) on the surface,
texel_rc (N,2) as image row/column, size.
CLI: python cat_uv_bake.py MESH SIZE OUT.npz   (for meshes whose colour field lives in another environment)
"""
import sys

import numpy as np
import open3d as o3d


def uv_bake(mesh_path, size=2048):
    legacy = o3d.io.read_triangle_mesh(mesh_path)
    tm = o3d.t.geometry.TriangleMesh.from_legacy(legacy)
    tm.compute_uvatlas(size=size, gutter=4.0)
    tri_uv = tm.triangle.texture_uvs.numpy().astype(np.float64)  # (F,3,2)
    faces = tm.triangle.indices.numpy().astype(np.int64)  # (F,3)
    verts = tm.vertex.positions.numpy().astype(np.float64)
    tm.vertex["ones"] = o3d.core.Tensor(np.ones((len(verts), 1), np.float32))
    baked = tm.bake_vertex_attr_textures(size, {"positions", "ones"}, margin=0.0, fill=0.0, update_material=False)
    posmap = baked["positions"].numpy()
    mask = baked["ones"].numpy()[..., 0] > 0.5

    # which way the baked image is stored: compare triangle centroids with the texel under their mean uv
    cen_uv, cen_xyz = tri_uv.mean(1), verts[faces].mean(1)
    pick = np.random.default_rng(0).choice(len(faces), min(2000, len(faces)), replace=False)
    col = np.clip((cen_uv[pick, 0] * size).astype(int), 0, size - 1)
    errs = {}
    for name, row in (("v_down", (cen_uv[pick, 1] * size).astype(int)), ("v_up", ((1 - cen_uv[pick, 1]) * size).astype(int))):
        row = np.clip(row, 0, size - 1)
        ok = mask[row, col]
        errs[name] = float(np.median(np.linalg.norm(posmap[row, col][ok] - cen_xyz[pick][ok], axis=1)))
    stored = min(errs, key=errs.get)
    if stored == "v_down":  # row 0 holds v = 0: flip so that row 0 holds v = 1, as three.js expects
        posmap, mask = posmap[::-1].copy(), mask[::-1].copy()
    print("uv atlas: %d faces, image %d px, texels on the surface %.1f%%, stored %s (centroid error %s)" % (
        len(faces), size, 100 * mask.mean(), stored, {k: round(v, 5) for k, v in errs.items()}), flush=True)

    # weld triangle corners that share a vertex and a texture coordinate
    corner_v = faces.reshape(-1)
    corner_uv = tri_uv.reshape(-1, 2)
    key = np.concatenate([corner_v[:, None], np.round(corner_uv * 1e6).astype(np.int64)], 1)
    _, first, inverse = np.unique(key, axis=0, return_index=True, return_inverse=True)
    pos = verts[corner_v[first]]
    uv = corner_uv[first]
    new_faces = inverse.reshape(-1, 3)
    rc = np.argwhere(mask)
    print("welded: %d vertices (from %d), %d faces" % (len(pos), len(verts), len(new_faces)), flush=True)
    return {"pos": pos, "faces": new_faces, "uv": uv, "texel_pos": posmap[mask], "texel_rc": rc, "size": size}


def fill_and_save(rgb_texels, rc, size, path, rounds=12):
    """paint the colours into the image, spread them a few pixels into the empty space (so seams do not show), save"""
    from PIL import Image
    img = np.zeros((size, size, 3), np.float32)
    w = np.zeros((size, size), np.float32)
    img[rc[:, 0], rc[:, 1]] = rgb_texels
    w[rc[:, 0], rc[:, 1]] = 1
    for _ in range(rounds):
        acc = np.zeros_like(img)
        cnt = np.zeros_like(w)
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            acc += np.roll(np.roll(img * w[..., None], dr, 0), dc, 1)
            cnt += np.roll(np.roll(w, dr, 0), dc, 1)
        grow = (w == 0) & (cnt > 0)
        img[grow] = acc[grow] / cnt[grow][:, None]
        w[grow] = 1
    Image.fromarray((np.clip(img, 0, 1) * 255).round().astype(np.uint8)).save(path, quality=90)


if __name__ == "__main__":
    out = uv_bake(sys.argv[1], int(sys.argv[2]))
    np.savez_compressed(sys.argv[3], **{k: v for k, v in out.items()})
