"""The real cat's fur at home (work/pets/PROTOCOL_CAT.md, "The real cat's fur at home"; demo only): move the Body tab's
photo coat (the C1 cat, painted from the real video frames, version 8) onto Agent-to-Sim's home mesh, which so far wears
a coat baked from its model's colour network (no fur detail). Both are the same cat, Pikachu. CPU only.

  - the C1 shape: of its canonical shape and its 92 posed frames, the one that best fits the home cat's rest pose after a
    rigid fit (one scale, rotation, shift), among those with four clear paws;
  - the home mesh (UV seams welded) is bent smoothly onto it, paws, nose and tail tip pinned (home2_retarget.nonrigid);
  - each home texel: its point on the bent surface -> the closest point on the C1 surface (exact point-to-triangle, so
    the photo coat's fur detail survives) -> the C1 photo coat there (bilinear);
  - fourth try (FADE=inward, the default): the fade runs only into the seen area, so unseen C1 colours never leak in
  - third try (C1_SHAPE=canonical): the C1 cat's own face is turned in frame 89 and straight in its canonical shape
  - SMEAR=cx,cy,cz,r_in,r_out (with HEAD_ONLY=1): the C1 coat's eye-like smear on the -x cheek replaced by a smooth
    fill from the fur round it ("The far-cheek smudge")
  - FILL=1 (fifth try, in place of HEAD_ONLY): the whole cat wears the photo coat; texels unseen by C1, far from its
    surface, or much darker than the fur within 5 cm (outside the front of the face) take a smooth fill from the reliable
    fur around them instead of the old coat ("The real fur on the whole cat at home")
  - BEND_SMOOTH=N: the bend's change smoothed over the mesh N rounds before the transfer (its crinkles otherwise
    bake folded streaks into the coat); SAVE_BENT=1 saves the bent mesh (bent.npz), BENT_ONLY=1 stops there
  - DIAG_W=1: also writes weights.jpg (R close to the C1 surface, G seen by C1's video, B head weight) and renders it
  - first try (v1): blended toward the home cat's current coat where no C1 frame saw that spot (C1's photo_cover),
    after a colour match; it turned those spots blue-white with hard edges, and the chosen C1 frame had its head turned
    (face on the side of the head). Second try: the C1 shape is chosen by the turn its head needs as well (the home cat's
    head fitted on its own); unseen spots get (a) C1's own colours there, as on the Body tab, or (b) the current coat,
    no colour match, fading over about 2 cm on the surface. Either way texels over 1 cm (fading from 0.5 cm) from the C1
    surface keep the current coat.

usage: python home_coat.py PAGE_DIR C1_UV_NPZ C1_PHOTO_TEX C1_COVER HOME_TEX OUT_DIR
  PAGE_DIR: a2s_cat.bin/.json (the home cat, with uv) and cat.bin/.json (the C1 cat's 92 posed frames, as on the page)
  C1_UV_NPZ: the C1 cat's UV mesh (canonical); C1_PHOTO_TEX: its photo coat; C1_COVER: photo_cover.jpg (black = no
  frame saw that texel); HOME_TEX: the home cat's current coat (a2s_cat_tex.jpg). Textures: row 0 holds v = 1.
Writes OUT_DIR/a2s_cat_tex_photo_c1.jpg and a2s_cat_tex_photo_home.jpg (forms (a) and (b); the home cat's UV layout,
2048 px), home_coat.json and home_coat_*.png (offline renders of the rest pose: current coat, (a), (b)).
"""
import json
import os
import sys

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt, map_coordinates, uniform_filter
from scipy.sparse import coo_matrix, diags
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from home2_coat import fill_and_save, rasterise, sample  # noqa: E402
from home2_retarget import landmarks, nonrigid  # noqa: E402
from home2_stand import weld  # noqa: E402
from home2_transfer import edges_of, icp, read_blobs  # noqa: E402

SIZE = 2048
HEAD = 0.10       # the head: within this of the nose, front to back
VOX = 0.01        # voxel size for fading the unseen spots over the surface
NEAR, FAR = 0.005, 0.010   # closest C1 point this far: photo coat in full / not at all
K_CAND = 12                # candidate C1 triangles per texel (nearest centroids)
DARK_FROM, DARK_FULL = 9.0, 15.0  # FILL: this many lightness points darker than the fur round it: starts / fully unreliable


def load_page(page_dir, name):
    m = json.load(open(os.path.join(page_dir, name + ".json")))
    return m, read_blobs(os.path.join(page_dir, name + ".bin"), m)


def rigid_fit(Q, P, tree_p):
    """the best of four starts (principal axes, signs) -> s, R, t, mean distance"""
    mp, mq = P.mean(0), Q.mean(0)
    ep = np.linalg.eigh(np.cov((P - mp).T))[1][:, ::-1]
    eq = np.linalg.eigh(np.cov((Q - mq).T))[1][:, ::-1]
    s0 = np.sqrt(((P - mp) ** 2).sum(1).mean() / ((Q - mq) ** 2).sum(1).mean())
    fits = []
    for sg in ([1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]):
        R0 = ep @ np.diag(sg) @ eq.T
        if np.linalg.det(R0) < 0:
            R0 = ep @ np.diag(np.array(sg) * [1, 1, -1]) @ eq.T
        fits.append(icp(Q, P, s0, R0, mp - s0 * R0 @ mq, tree_p))
    return min(fits, key=lambda f: f[3])[:4]


def closest_on_triangles(p, a, b, c):
    """closest point of triangle (a, b, c) to p, all (N,3) -> barycentric weights (N,3) (Ericson, Real-Time Collision
    Detection 5.1.5)"""
    ab, ac, ap = b - a, c - a, p - a
    d1, d2 = (ab * ap).sum(1), (ac * ap).sum(1)
    bp = p - b
    d3, d4 = (ab * bp).sum(1), (ac * bp).sum(1)
    cp = p - c
    d5, d6 = (ab * cp).sum(1), (ac * cp).sum(1)
    va, vb, vc = d3 * d6 - d5 * d4, d5 * d2 - d1 * d6, d1 * d4 - d3 * d2
    den = np.where(np.abs(va + vb + vc) < 1e-20, 1e-20, va + vb + vc)
    v, w = vb / den, vc / den
    bary = np.stack([1 - v - w, v, w], 1)  # inside the face
    with np.errstate(divide="ignore", invalid="ignore"):
        # edges and corners: Ericson tests a, b, ab, c, ac, bc in that order and returns at the first that holds, so
        # here they are applied in reverse (an earlier test overwrites a later one)
        m = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)  # edge bc
        t = np.clip((d4 - d3) / ((d4 - d3) + (d5 - d6)), 0, 1)
        bary[m] = np.stack([np.zeros(m.sum()), 1 - t[m], t[m]], 1)
        m = (vb <= 0) & (d2 >= 0) & (d6 <= 0)  # edge ac
        t = np.clip(d2 / (d2 - d6), 0, 1)
        bary[m] = np.stack([1 - t[m], np.zeros(m.sum()), t[m]], 1)
        m = (d6 >= 0) & (d5 <= d6)  # corner c
        bary[m] = [0, 0, 1]
        m = (vc <= 0) & (d1 >= 0) & (d3 <= 0)  # edge ab
        t = np.clip(d1 / (d1 - d3), 0, 1)
        bary[m] = np.stack([1 - t[m], t[m], np.zeros(m.sum())], 1)
        m = (d3 >= 0) & (d4 <= d3)  # corner b
        bary[m] = [0, 1, 0]
        m = (d1 <= 0) & (d2 <= 0)  # corner a
        bary[m] = [1, 0, 0]
    return bary


def closest_on_mesh(pts, X, F, chunk=60000):
    """closest point on the mesh (X, F) to each of pts -> face index, barycentric weights, distance"""
    cen = X[F].mean(1)
    tree = cKDTree(cen)
    fi_out, bary_out, d_out = np.zeros(len(pts), np.int64), np.zeros((len(pts), 3)), np.zeros(len(pts))
    for s in range(0, len(pts), chunk):
        p = pts[s:s + chunk]
        _, cand = tree.query(p, k=K_CAND)
        pr = np.repeat(p, K_CAND, 0)
        f = cand.ravel()
        a, b, c = X[F[f, 0]], X[F[f, 1]], X[F[f, 2]]
        bc = closest_on_triangles(pr, a, b, c)
        q = bc[:, :1] * a + bc[:, 1:2] * b + bc[:, 2:] * c
        d = np.linalg.norm(q - pr, axis=1).reshape(-1, K_CAND)
        k = np.argmin(d, 1)
        r = np.arange(len(p)) * K_CAND + k
        fi_out[s:s + chunk], bary_out[s:s + chunk], d_out[s:s + chunk] = f[r], bc[r], d[np.arange(len(p)), k]
    return fi_out, bary_out, d_out


def lightness(rgb):
    """CIELAB L of sRGB colours in 0..1 (per colour), as a2s_cat_export.py measured the coat"""
    lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    y = lin @ np.array([0.2126729, 0.7151522, 0.0721750])
    f = np.where(y > (6 / 29) ** 3, np.maximum(y, 1e-8) ** (1 / 3), y / (3 * (6 / 29) ** 2) + 4 / 29)
    return 116 * f - 16


def render(X, F, UV, tex, view, size=700, crop=None):
    """a plain z-buffered render with the texture, orthographic; view: 'front', 'side', 'top' (the home cat's frame:
    y down, z towards the tail). crop: (vertex mask) to frame only part of the cat"""
    if view == "other_side":  # the -x side: turned half a turn about the vertical
        X, view = X * np.array([-1.0, 1.0, -1.0]), "side"
    elif view == "three_quarter":  # from the front, turned 40 degrees towards +x
        a = np.radians(40)
        X, view = X @ np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]]).T, "front"
    axes = {"front": (0, 1, 2, 1), "side": (2, 1, 0, -1), "top": (0, 2, 1, 1)}[view]
    V = np.stack([X[:, axes[0]], X[:, axes[1]], axes[3] * X[:, axes[2]]], 1)
    sel = V if crop is None else V[crop]
    lo, hi = sel[:, :2].min(0), sel[:, :2].max(0)
    sc = (size - 20) / (hi - lo).max()
    S = np.c_[(V[:, :2] - lo) * sc + 10, V[:, 2]]
    img = np.full((size, size, 3), 0.12)
    zb = np.full((size, size), np.inf)
    n = np.cross(X[F[:, 1]] - X[F[:, 0]], X[F[:, 2]] - X[F[:, 0]])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    vdir = np.zeros(3)
    vdir[axes[2]] = 1
    shade = 0.35 + 0.65 * np.abs(n @ vdir)
    for f in range(len(F)):
        a, b, c = S[F[f]]
        x0, x1 = int(max(np.floor(min(a[0], b[0], c[0])), 0)), int(min(np.ceil(max(a[0], b[0], c[0])), size - 1))
        y0, y1 = int(max(np.floor(min(a[1], b[1], c[1])), 0)), int(min(np.ceil(max(a[1], b[1], c[1])), size - 1))
        if x1 < x0 or y1 < y0:
            continue
        xs, ys = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5)
        d = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
        if abs(d) < 1e-12:
            continue
        l1 = ((b[1] - c[1]) * (xs - c[0]) + (c[0] - b[0]) * (ys - c[1])) / d
        l2 = ((c[1] - a[1]) * (xs - c[0]) + (a[0] - c[0]) * (ys - c[1])) / d
        l3 = 1 - l1 - l2
        ok = (l1 >= 0) & (l2 >= 0) & (l3 >= 0)
        if not ok.any():
            continue
        z = l1 * a[2] + l2 * b[2] + l3 * c[2]
        yy, xx = (ys[ok] - 0.5).astype(int), (xs[ok] - 0.5).astype(int)
        nearer = z[ok] < zb[yy, xx]
        if not nearer.any():
            continue
        yy, xx = yy[nearer], xx[nearer]
        L = np.stack([l1[ok][nearer], l2[ok][nearer], l3[ok][nearer]], 1)
        zb[yy, xx] = z[ok][nearer]
        img[yy, xx] = sample(tex, L @ UV[F[f]]) * shade[f]
    return img


def smear_fill(photo, tpos, spec, vox=0.003, iters=400):
    """spec (cx, cy, cz, r_in, r_out), rest-pose metres: within r_in of the centre the photo colours are replaced by a
    smooth fill from the ring outside r_in (3 mm voxels; each inner voxel repeatedly set to its occupied neighbours'
    mean), fading back to the photo by r_out. Returns the new per-texel photo colours."""
    c, r_in, r_out = np.array(spec[:3]), spec[3], spec[4]
    d = np.linalg.norm(tpos - c, axis=1)
    sel = np.where(d < r_out + 0.012)[0]
    ijk = np.floor((tpos[sel] - (c - r_out - 0.02)) / vox).astype(int)
    shape = tuple(ijk.max(0) + 2)
    acc, cnt, inner = np.zeros(shape + (3,)), np.zeros(shape), np.zeros(shape, bool)
    known = d[sel] >= r_in
    np.add.at(acc, tuple(ijk[known].T), photo[sel[known]])
    np.add.at(cnt, tuple(ijk[known].T), 1)
    occ = np.zeros(shape, bool)
    occ[tuple(ijk.T)] = True
    inner[tuple(ijk[~known].T)] = True
    inner &= cnt == 0  # voxels with any ring texel stay as measured
    val = np.where(cnt[..., None] > 0, acc / np.maximum(cnt, 1)[..., None], 0)
    val[inner] = val[occ & ~inner].mean(0)  # start from the ring's mean colour
    offs = [o for o in np.ndindex(3, 3, 3) if o != (1, 1, 1)]
    pad = lambda a: np.pad(a, [(1, 1)] * 3 + [(0, 0)] * (a.ndim - 3))  # noqa: E731
    occp = pad(occ).astype(float)
    for _ in range(iters):
        vp = pad(val)
        s = sum(vp[o[0]:o[0] + shape[0], o[1]:o[1] + shape[1], o[2]:o[2] + shape[2]] * occp[o[0]:o[0] + shape[0], o[1]:o[1] + shape[1], o[2]:o[2] + shape[2], None] for o in offs)
        n = sum(occp[o[0]:o[0] + shape[0], o[1]:o[1] + shape[1], o[2]:o[2] + shape[2]] for o in offs)
        val[inner] = (s / np.maximum(n, 1)[..., None])[inner]
    have = occ.copy()  # spread into the empty voxels next to the surface, so trilinear reads never mix in zeros
    for _ in range(2):
        vp, hp = pad(val), pad(have).astype(float)
        s = sum(vp[o[0]:o[0] + shape[0], o[1]:o[1] + shape[1], o[2]:o[2] + shape[2]] * hp[o[0]:o[0] + shape[0], o[1]:o[1] + shape[1], o[2]:o[2] + shape[2], None] for o in offs)
        n = sum(hp[o[0]:o[0] + shape[0], o[1]:o[1] + shape[1], o[2]:o[2] + shape[2]] for o in offs)
        new = ~have & (n > 0)
        val[new] = (s / np.maximum(n, 1)[..., None])[new]
        have |= new
    q = ((tpos[sel] - (c - r_out - 0.02)) / vox - 0.5).T
    fill = np.stack([map_coordinates(val[..., k], q, order=1, mode="nearest") for k in range(3)], 1)
    t = np.clip((r_out - d[sel]) / (r_out - r_in), 0, 1)
    w = t * t * (3 - 2 * t)
    out = photo.copy()
    out[sel] = (1 - w[:, None]) * photo[sel] + w[:, None] * fill
    print("smear fill: %d texels replaced in full, %d faded; %d inner voxels filled from %d" % (
        (w > 0.99).sum(), ((w > 0.01) & (w <= 0.99)).sum(), inner.sum(), (occ & ~inner).sum()), flush=True)
    return out


def _neighbour_mean(val, weight, offs):
    """per voxel: the weighted mean of its 26 neighbours (weight 0 outside the grid); also the summed weight"""
    shape = weight.shape
    vp = np.pad(val, [(1, 1)] * 3 + [(0, 0)])
    wp = np.pad(weight, [(1, 1)] * 3)
    s, n = np.zeros(val.shape), np.zeros(shape)
    for o in offs:
        w = wp[o[0]:o[0] + shape[0], o[1]:o[1] + shape[1], o[2]:o[2] + shape[2]]
        s += vp[o[0]:o[0] + shape[0], o[1]:o[1] + shape[1], o[2]:o[2] + shape[2]] * w[..., None]
        n += w
    return s / np.maximum(n, 1e-9)[..., None], n


def voxel_fill(photo, tpos, rel, vox=0.004, iters=300):
    """FILL: a smooth colour for every texel from the reliable ones around it. Voxels whose texels are mostly reliable
    (mean rel >= 0.5) keep their rel-weighted photo colour; the other occupied voxels start from the nearest reliable
    voxel and are then repeatedly set to the mean of their occupied neighbours (a fill over the surface). Read back
    per texel with trilinear interpolation."""
    lo = tpos.min(0) - 2 * vox
    ijk = np.floor((tpos - lo) / vox).astype(int)
    shape = tuple(ijk.max(0) + 3)
    idx = tuple(ijk.T)
    acc, wsum, cnt = np.zeros(shape + (3,)), np.zeros(shape), np.zeros(shape)
    np.add.at(acc, idx, rel[:, None] * photo)
    np.add.at(wsum, idx, rel)
    np.add.at(cnt, idx, 1)
    occ = cnt > 0
    known = occ & (wsum >= 0.5 * cnt) & (wsum > 1e-6)
    assert known.any(), "no reliable voxel"
    val = np.where(known[..., None], acc / np.maximum(wsum, 1e-9)[..., None], 0)
    unk = occ & ~known
    _, ind = distance_transform_edt(~known, return_indices=True)
    val[unk] = val[ind[0][unk], ind[1][unk], ind[2][unk]]
    offs = [o for o in np.ndindex(3, 3, 3) if o != (1, 1, 1)]
    occf = occ.astype(float)
    for _ in range(iters):
        m, _n = _neighbour_mean(val, occf, offs)
        val[unk] = m[unk]
    have = occ.copy()  # spread one voxel into the empty space so trilinear reads never mix in zeros
    for _ in range(2):
        m, n = _neighbour_mean(val, have.astype(float), offs)
        new = ~have & (n > 0)
        val[new] = m[new]
        have |= new
    q = ((tpos - lo) / vox - 0.5).T
    print("voxel fill: %d occupied voxels, %d reliable, %d filled" % (occ.sum(), known.sum(), unk.sum()), flush=True)
    return np.stack([map_coordinates(val[..., k], q, order=1, mode="nearest") for k in range(3)], 1)


def main():
    page_dir, c1_npz, c1_tex_path, c1_cover_path, home_tex_path, out = sys.argv[1:7]
    os.makedirs(out, exist_ok=True)
    am, A = load_page(page_dir, "a2s_cat")
    P, F, UV = A["pos"].astype(np.float64), A["faces"].astype(np.int64), A["uv"].astype(np.float64)
    cm, C = load_page(page_dir, "cat")
    z = np.load(c1_npz)
    Qc, Fc, UVc = z["pos"].astype(np.float64), z["faces"].astype(np.int64), z["uv"].astype(np.float64)
    nfr = cm["frames"]
    lo, hi = np.array(cm["lo"]), np.array(cm["hi"])
    frames = lo + C["pos"].reshape(nfr, -1, 3).astype(np.float64) / 65535.0 * (hi - lo)
    assert frames.shape[1] == len(Qc) and np.array_equal(C["faces"].reshape(-1, 3).astype(np.int64), Fc), "C1 meshes differ"
    tex_c1 = np.asarray(Image.open(c1_tex_path).convert("RGB"), dtype=np.float32) / 255.0
    cover = np.asarray(Image.open(c1_cover_path).convert("L"), dtype=np.float32)[..., None] / 255.0
    tex_home = np.asarray(Image.open(home_tex_path).convert("RGB"), dtype=np.float32) / 255.0
    print("home cat: %d vertices, %d faces; C1 cat: %d vertices, %d posed frames" % (len(P), len(F), len(Qc), nfr), flush=True)

    # ---- the C1 shape that best fits the home cat's rest pose, among those with four clear paws
    tree_p = cKDTree(P)
    widc, _, _ = weld(Qc, Fc)  # seam copies joined (paw search and bend need the surface in one piece)
    firstc = np.zeros(widc.max() + 1, np.int64)
    firstc[widc] = np.arange(len(widc))
    Fcw = widc[Fc]
    cands = [("canonical", Qc)] + [("frame %d" % k, frames[k]) for k in range(nfr)]
    scored = []
    for name, X in cands:
        s, R, t, fd = rigid_fit(X[::6], P[::3], cKDTree(P[::3]))
        scored.append((fd, name, s, R, t))
    scored.sort(key=lambda x: x[0])
    nose = P[np.argsort(P[:, 2])[:20]].mean(0)
    headP = P[P[:, 2] < P[:, 2].min() + HEAD]
    rows = []
    for fd, name, s, R, t in scored[:40]:
        X = dict(cands)[name]
        Xa = s * X @ R.T + t
        try:
            landmarks(Xa[firstc], Fcw)
        except AssertionError:
            print("  C1 %-10s fit %.1f mm, four paws no" % (name, 1000 * fd), flush=True)
            continue
        # the C1 cat's head: the same front slab (2 cm deeper), near the home cat's nose
        near_head = Xa[(Xa[:, 2] < P[:, 2].min() + HEAD + 0.02) & (np.linalg.norm(Xa - nose, axis=1) < 1.6 * HEAD)]
        _, Rh, _, hd = icp(headP[::2], near_head, 1.0, np.eye(3), np.zeros(3), cKDTree(near_head))[:4]
        turn = float(np.degrees(np.arccos(np.clip((np.trace(Rh) - 1) / 2, -1, 1))))
        rows.append((turn, fd, name, Xa))
        print("  C1 %-10s fit %.1f mm, four paws yes, head needs %.0f deg more (head fit %.1f mm)" % (name, 1000 * fd, turn, 1000 * hd), flush=True)
    assert rows, "no C1 shape with four clear paws among the 40 best fits"
    best_turn = min(r[0] for r in rows)
    turn, fd, name, Xa = min([r for r in rows if r[0] <= best_turn + 5], key=lambda r: r[1])
    if os.environ.get("C1_SHAPE"):  # third try: the head test could not see the turn; the canonical shape faces straight
        turn, fd, name, Xa = [r for r in rows if r[2] == os.environ["C1_SHAPE"]][0]
    chosen = (name, Xa, fd)
    print("chosen: %s (head turn %.0f deg)" % (name, turn), flush=True)
    name, Xc, fd0 = chosen
    s, R, t, fd = rigid_fit(dict(cands)[name], P, tree_p)  # refit on all points
    Xc = s * dict(cands)[name] @ R.T + t
    print("C1 shape: %s; rigid fit %.1f mm (best of all %.1f, median %.1f)" % (
        name, 1000 * fd, 1000 * scored[0][0], 1000 * np.median([x[0] for x in scored])), flush=True)

    # ---- bend the home mesh (seams welded) onto the C1 shape, paws, nose and tail tip pinned
    wid, Pw, _ = weld(P, F)
    Fw = wid[F]
    keep = Fw[:, 0] != Fw[:, 1]
    keep &= (Fw[:, 1] != Fw[:, 2]) & (Fw[:, 0] != Fw[:, 2])
    Xb_w = nonrigid(Pw, Fw[keep], Xc[firstc], Fcw, cKDTree(Xc[firstc]), [30.0, 10.0, 3.0])
    if int(os.environ.get("BEND_SMOOTH", "0")) > 0:  # "The head": the bend's crinkles smoothed out of the change
        n_it, ew = int(os.environ["BEND_SMOOTH"]), edges_of(Fw[keep])
        Adj = coo_matrix((np.ones(2 * len(ew)), (np.r_[ew[:, 0], ew[:, 1]], np.r_[ew[:, 1], ew[:, 0]])), shape=(len(Pw),) * 2).tocsr()
        Mavg = diags(1 / np.maximum(np.asarray(Adj.sum(1)).ravel(), 1)) @ Adj
        Dw = Xb_w - Pw
        for _ in range(n_it):
            Dw = 0.5 * Dw + 0.5 * (Mavg @ Dw)
        moved = np.linalg.norm(Pw + Dw - Xb_w, axis=1)
        print("bend smoothed (%d rounds): moved median %.1f mm, p99 %.1f mm" % (n_it, 1000 * np.median(moved), 1000 * np.percentile(moved, 99)), flush=True)
        Xb_w = Pw + Dw
    Pb = Xb_w[wid]
    print("home mesh: %d vertices welded to %d; bent onto C1" % (len(P), len(Pw)), flush=True)
    if os.environ.get("SAVE_BENT") == "1":  # "The head": the home mesh bent onto C1, and C1 fitted, in the home rest frame
        np.savez(os.path.join(out, "bent.npz"), P=P, Pb=Pb, Xc=Xc, Fc=Fc, UVc=UVc, nose=nose)
        print("saved bent.npz", flush=True)
        if os.environ.get("BENT_ONLY") == "1":
            return

    # ---- every home texel -> closest C1 point -> photo coat
    tpos, trc = rasterise(Pb, F, UV, SIZE)
    print("home texels: %d (%.1f%% of the image)" % (len(tpos), 100 * len(tpos) / SIZE ** 2), flush=True)
    fi, bc, dist = closest_on_mesh(tpos, Xc, Fc)
    uvq = (bc[:, :, None] * UVc[Fc[fi]]).sum(1)
    photo = sample(tex_c1, uvq)
    seen = np.clip((sample(cover, uvq)[:, 0] - 0.02) / 0.08, 0, 1)
    near = np.clip((FAR - dist) / (FAR - NEAR), 0, 1)
    home = tex_home[trc[:, 0], trc[:, 1]]
    # (b): the unseen spots fade over about 2 cm on the surface: the seen weight averaged in 1 cm voxels, a 3-voxel box,
    # read back with trilinear interpolation
    lo3 = tpos.min(0) - 2 * VOX
    ijk = np.floor((tpos - lo3) / VOX).astype(int)
    shape = tuple(ijk.max(0) + 3)
    acc, cnt = np.zeros(shape), np.zeros(shape)
    np.add.at(acc, tuple(ijk.T), seen)
    np.add.at(cnt, tuple(ijk.T), 1)
    acc, cnt = uniform_filter(acc, 3), uniform_filter(cnt, 3)
    grid = np.where(cnt > 1e-9, acc / np.maximum(cnt, 1e-9), 0)
    seen_soft = map_coordinates(grid, ((tpos - lo3) / VOX - 0.5).T, order=1, mode="nearest")
    if os.environ.get("FADE", "inward") == "inward":  # fourth try: no leak of unseen C1 colours across the edge
        ramp = np.clip(2 * seen_soft - 1, 0, 1)
        forms = {"c1": near, "home": seen * ramp * ramp * (3 - 2 * ramp) * near}
    else:  # second and third tries: the fade blurred outward as well
        forms = {"c1": near, "home": np.clip(seen_soft, 0, 1) * near}
    head_w = np.ones(len(tpos))
    if os.environ.get("HEAD_ONLY") == "1":  # "A real face for the cat at home": the photo coat on the head only
        tpos0, trc0 = rasterise(P, F, UV, SIZE)  # the same texels on the home cat's own rest pose
        assert np.array_equal(trc0, trc)
        back = tpos0[:, 2] - P[:, 2].min()  # behind the nose (z towards the tail)
        below = tpos0[:, 1] - nose[1]       # below the nose (y down)
        sm = lambda x: x * x * (3 - 2 * x)  # noqa: E731
        head_w = sm(np.clip((0.10 - back) / 0.03, 0, 1)) * sm(np.clip((0.07 - below) / 0.03, 0, 1))
        forms = {k: w * head_w for k, w in forms.items()}
        print("head only: %.1f%% of texels fully head, %.1f%% partly" % (100 * (head_w > 0.99).mean(), 100 * ((head_w > 0.01) & (head_w <= 0.99)).mean()), flush=True)
        if os.environ.get("SMEAR"):  # "The far-cheek smudge": the photo coat's eye-like smear filled with the fur round it
            photo = smear_fill(photo, tpos0, [float(v) for v in os.environ["SMEAR"].split(",")])
    whole = os.environ.get("FILL") == "1"
    if whole:  # "The real fur on the whole cat at home" (fifth try): unreliable texels filled from the fur around them
        tpos0, trc0 = rasterise(P, F, UV, SIZE)
        assert np.array_equal(trc0, trc)
        if os.environ.get("SMEAR"):
            photo = smear_fill(photo, tpos0, [float(v) for v in os.environ["SMEAR"].split(",")])
        rel0 = forms["home"]  # seen by C1's video (inward fade) and close to its surface
        Lp = lightness(photo)
        lo1 = tpos0.min(0) - 3 * VOX
        ij1 = np.floor((tpos0 - lo1) / VOX).astype(int)
        sh1 = tuple(ij1.max(0) + 4)
        accL, cntL = np.zeros(sh1), np.zeros(sh1)
        np.add.at(accL, tuple(ij1.T), rel0 * Lp)
        np.add.at(cntL, tuple(ij1.T), rel0)
        accL, cntL = uniform_filter(accL, 5), uniform_filter(cntL, 5)  # the reliable fur within about 5 cm
        loc = map_coordinates(np.where(cntL > 1e-9, accL / np.maximum(cntL, 1e-9), 0), ((tpos0 - lo1) / VOX - 0.5).T,
                              order=1, mode="nearest")
        dark = np.clip((loc - Lp - DARK_FROM) / (DARK_FULL - DARK_FROM), 0, 1)
        face_front = tpos0[:, 2] - P[:, 2].min() < 0.03
        dark[face_front] = 0  # eyes, nose and mouth keep their dark marks
        rel = rel0 * (1 - dark)
        fill = voxel_fill(photo, tpos0, rel)
        photo = rel[:, None] * photo + (1 - rel[:, None]) * fill
        forms = {"fill": np.ones(len(tpos))}
        L_after = lightness(photo)
        fill_info = {"share_rel_under_half": float((rel < 0.5).mean()), "share_dark_over_half": float((dark > 0.5).mean()),
                     "share_L_under_35_outside_face_before": float(((Lp < 35) & ~face_front).mean()),
                     "share_L_under_35_outside_face_after": float(((L_after < 35) & ~face_front).mean())}
        print("whole-cat fill: %s" % json.dumps(fill_info), flush=True)
        if os.environ.get("DIAG_W") == "1":  # R close to C1, G reliable (seen and close), B not too dark
            fill_and_save(np.stack([near, rel0, 1 - dark], 1), trc, SIZE, os.path.join(out, "weights.jpg"))
    elif os.environ.get("DIAG_W") == "1":  # why a texel keeps the old coat: R close to C1, G seen by C1's video, B head weight
        fade = forms["home"] / np.maximum(near * head_w, 1e-9)
        fill_and_save(np.stack([near, np.clip(fade, 0, 1), head_w], 1), trc, SIZE, os.path.join(out, "weights.jpg"))
    info = {"c1_shape": name, "head_turn_deg": turn, "rigid_fit_mm": 1000 * fd, "texels": int(len(tpos)),
            "closest_c1_mm_median": 1000 * float(np.median(dist)), "closest_c1_mm_p90": 1000 * float(np.percentile(dist, 90)),
            "closest_c1_mm_p99": 1000 * float(np.percentile(dist, 99)),
            "share_unseen_by_c1": float((seen < 0.01).mean()), "share_too_far": float((near < 0.01).mean()),
            "lightness_L_current_mean": float(lightness(home).mean()), "real_cat_L": 74.0}
    for k, w in forms.items():
        rgb = w[:, None] * photo + (1 - w[:, None]) * home
        fill_and_save(rgb, trc, SIZE, os.path.join(out, "a2s_cat_tex_photo_%s.jpg" % k))
        L = float(lightness(rgb).mean())
        info[k] = {"share_photo_full": float((w > 0.99).mean()), "share_photo_none": float((w < 0.01).mean()),
                   "lightness_L_mean": L, "readout_lightness": "within 4 of 74" if abs(L - 74) <= 4 else "off by more than 4"}
    if whole:
        info["whole_cat_fill"] = fill_info
    json.dump(info, open(os.path.join(out, "home_coat.json"), "w"), indent=1)
    print(json.dumps(info, indent=1), flush=True)

    # ---- offline renders of the rest pose: current coat against photo coat
    texs = [tex_home] + [np.asarray(Image.open(os.path.join(out, "a2s_cat_tex_photo_%s.jpg" % k)).convert("RGB"),
                                    dtype=np.float32) / 255.0 for k in forms]
    if os.environ.get("DIAG_W") == "1":
        texs.append(np.asarray(Image.open(os.path.join(out, "weights.jpg")).convert("RGB"), dtype=np.float32) / 255.0)
    head = (P[:, 2] < P[:, 2].min() + 0.12) & (P[:, 1] < nose[1] + 0.05)
    views = (("front", None), ("side", None), ("top", None), ("front", head), ("side", head))
    later = os.environ.get("HEAD_ONLY") == "1" or whole
    if os.environ.get("HEAD_ONLY") == "1":
        views = (("front", head), ("side", head), ("other_side", head), ("three_quarter", head), ("side", None))
    if whole:
        views = (("front", head), ("side", head), ("other_side", head), ("three_quarter", head), ("side", None),
                 ("other_side", None), ("top", None))
    for view, crop in views:
        tag = view + ("_head" if crop is not None else "") + ("_whole" if crop is None and later and view != "top" else "")
        row = [render(P, F, UV, tx, view, crop=crop) for tx in texs]
        Image.fromarray((np.concatenate(row, 1) * 255).clip(0, 255).astype(np.uint8)).save(
            os.path.join(out, "home_coat_%s.png" % tag))
        print("render %s done" % tag, flush=True)


if __name__ == "__main__":
    main()
