"""Graft the C1 cat's own head (Body tab mesh and photo coat) onto the home cat ("The head", PROTOCOL_CAT.md). CPU.

The C1 cat is moved rigidly so its head sits on the home head (the same move as head_reshape.py) and its head is cut
off behind the ears (CUT behind the nose tip) and below the chin (CHIN below it). Second round ("finishing the graft"):
  - the cut is exact: triangles crossing a cut plane are clipped there, so the open edge is a smooth line instead of
    following C1's triangles (the first round kept only whole triangles: a zigzag edge);
  - the home body near the head is stretched in one smooth piece (sparse least squares on the welded mesh: the
    displacement's Laplacian, FIT_A for sitting DEPTH beneath the piece along its normal from BAND_OUT outside the cut
    to INNER0 inside, deeper further in, and a weak pull to a tucked place towards the head centre past INNER1, where
    the home ears and chin go; no move at BLEND outside the cut). Moving each vertex to its own closest point (g8)
    crumpled the neck into spikes at the throat; smoothing that and blending outward (g9) still crinkled it;
  - the piece's rim (RIM from its edge) is laid onto the fitted body (LIFT above it), so its edge has no step; under
    the chin, where the head's edge stands more than RIM_GAP clear of the neck (the jowls overhang it), it is left
    hanging instead: dragging it onto the neck creased the throat (g10, g11);
  - the head piece is skinned from the nearest 4 fitted body vertices, so its edge moves with the body under it;
  - its coat (a copy of the C1 photo coat, a2s_graft_tex.jpg) fades into the body's over FADE from the open edge; with
    COVER (C1's photo_cover.jpg), spots C1's video never saw or much darker than the fur within 5 cm (outside the
    front 3 cm of the face) take a smooth fill from the fur around them first, as the whole-cat coat does (FILL).
Writes graft.npz, the page's files a2s_graft.bin/.json (+ a2s_graft_tex.jpg) and offline renders.
usage: python head_graft.py TOOLS_DIR PAGE_DIR BENT_NPZ HOME_TEX C1_TEX OUT_DIR
env: CUT, CHIN, BLEND, DEPTH, FADE, RIM, RIM_GAP, BAND_OUT (metres), FIT_A, FIT_B, FIT_ITERS, COVER (path);
FIT_ONLY=1 stops before the renders. The page's files: the defaults with COVER (experiments_head/g15)."""
import json
import os
import sys
sys.path.insert(0, sys.argv[1])
import numpy as np
from PIL import Image
from scipy.sparse import block_diag, coo_matrix, diags
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree
from home_coat import DARK_FROM, DARK_FULL, closest_on_mesh, lightness, load_page, render, voxel_fill
from scipy.ndimage import map_coordinates, uniform_filter
from home2_coat import rasterise, sample
from home2_transfer import lbs, pack

page, bent, home_tex, c1_tex, out = sys.argv[2:7]
CUT = float(os.environ.get("CUT", 0.095))
CHIN = float(os.environ.get("CHIN", 0.075))
BLEND = float(os.environ.get("BLEND", 0.04))
DEPTH = float(os.environ.get("DEPTH", 0.002))
FADE = float(os.environ.get("FADE", 0.015))
RIM = float(os.environ.get("RIM", 0.008))   # the piece bends down onto the body over this from its edge
RIM_GAP = float(os.environ.get("RIM_GAP", 0.006))  # ... where its edge is within this of the body
LIFT = 0.0007  # ... ending this far above it
FIT_A = float(os.environ.get("FIT_A", 0.3))   # weight of sitting beneath the piece (along its normal)
FIT_G = 0.01 * FIT_A                          # ... and of staying near the closest point (no sliding)
FIT_B = float(os.environ.get("FIT_B", 0.01))  # weight of the tucked place deep inside
FIT_ITERS = int(os.environ.get("FIT_ITERS", 4))
BAND_OUT = float(os.environ.get("BAND_OUT", 0.0))  # how far outside the cut vertices are also held beneath the piece
INNER0, INNER1 = 0.015, 0.03
COVER = os.environ.get("COVER")  # C1's photo_cover.jpg: with it, unseen and dark spots on the head are filled
SINK = 0.3   # inside the cut the body sinks this much deeper per metre inside
TUCK = 0.5
os.makedirs(out, exist_ok=True)
am, A = load_page(page, "a2s_cat")
F, UV = A["faces"].astype(np.int64), A["uv"].astype(np.float64)
z = np.load(bent)
P, Pb, Xc, Fc, UVc, nose = z["P"], z["Pb"], z["Xc"], z["Fc"].astype(np.int64), z["UVc"], z["nose"]
T = np.asarray(Image.open(home_tex).convert("RGB"), dtype=np.float32) / 255.0
Tc = np.asarray(Image.open(c1_tex).convert("RGB"), dtype=np.float32) / 255.0
sm = lambda x: x * x * (3 - 2 * x)  # noqa: E731
zmin = P[:, 2].min()


def inside(X):
    """metres inside the head piece's region (behind the nose less than CUT, below it less than CHIN); < 0 outside"""
    return np.minimum(CUT - (X[:, 2] - zmin), CHIN - (X[:, 1] - nose[1]))


back, below = P[:, 2] - zmin, P[:, 1] - nose[1]
w = sm(np.clip((0.12 - back) / 0.04, 0, 1)) * sm(np.clip((0.09 - below) / 0.04, 0, 1))
full = w > 0.99

# ---- the C1 cat moved so its head sits on the home head
mp, mb = P[full].mean(0), Pb[full].mean(0)
U, s, Vt = np.linalg.svd((Pb[full] - mb).T @ (P[full] - mp))
R = (U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt).T
Xh = (Xc - mb) @ R.T + mp
cen = P[full].mean(0)


def weld_ids(X):
    _, wid = np.unique(np.round(X * 1e6).astype(np.int64), axis=0, return_inverse=True)
    return wid.ravel()


def vertex_normals(X, Fx):
    """area-weighted normals, shared across UV seams (welded by position), pointing away from the mesh centre"""
    wid = weld_ids(X)
    fn = np.cross(X[Fx[:, 1]] - X[Fx[:, 0]], X[Fx[:, 2]] - X[Fx[:, 0]])
    acc = np.zeros((wid.max() + 1, 3))
    for k in range(3):
        np.add.at(acc, wid[Fx[:, k]], fn)
    n = acc[wid]
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    if np.mean(np.sum(n * (X - X.mean(0)), 1)) < 0:
        n = -n
    return n


Nh = vertex_normals(Xh, Fc)


def clip(X, UVx, Fx, d):
    """the part of the mesh where d >= 0, cut exactly at d = 0 (new vertices on cut edges, shared by both faces)"""
    d = np.where(np.abs(d) < 2e-4, 0.0, d)
    pos = d[Fx] >= 0
    keep = Fx[pos.all(1)]
    mixed = Fx[pos.any(1) & ~pos.all(1)]
    newX, newU, newv, faces = [], [], {}, [keep]

    def cut(i, j):
        key = (min(i, j), max(i, j))
        if key not in newv:
            t = d[i] / (d[i] - d[j])
            newX.append(X[i] + t * (X[j] - X[i]))
            newU.append(UVx[i] + t * (UVx[j] - UVx[i]))
            newv[key] = len(X) + len(newX) - 1
        return newv[key]

    extra = []
    for f in mixed:
        poly = []
        for k in range(3):
            i, j = f[k], f[(k + 1) % 3]
            if d[i] >= 0:
                poly.append(i)
            if (d[i] > 0 > d[j]) or (d[i] < 0 < d[j]):
                poly.append(cut(i, j))
        for k in range(1, len(poly) - 1):
            extra.append([poly[0], poly[k], poly[k + 1]])
    if extra:
        faces.append(np.array(extra, np.int64))
    Fn = np.concatenate(faces)
    Xn = np.concatenate([X, np.array(newX).reshape(-1, 3)])
    Un = np.concatenate([UVx, np.array(newU).reshape(-1, 2)])
    used = np.unique(Fn)
    remap = -np.ones(len(Xn), np.int64)
    remap[used] = np.arange(len(used))
    return Xn[used], Un[used], remap[Fn]


# ---- the head piece: clipped at the two planes
H, UVh, Fh = Xh, UVc.astype(np.float64), Fc
H, UVh, Fh = clip(H, UVh, Fh, CUT - (H[:, 2] - zmin))
H, UVh, Fh = clip(H, UVh, Fh, CHIN - (H[:, 1] - nose[1]))
# drop slivers (zero-area faces from cuts through a vertex)
area = np.linalg.norm(np.cross(H[Fh[:, 1]] - H[Fh[:, 0]], H[Fh[:, 2]] - H[Fh[:, 0]]), axis=1)
Fh = Fh[area > 1e-12]
wid = weld_ids(H)
E = np.sort(np.concatenate([wid[Fh][:, [0, 1]], wid[Fh][:, [1, 2]], wid[Fh][:, [2, 0]]]), 1)
Eu, ec = np.unique(E, axis=0, return_counts=True)
rep = np.zeros(wid.max() + 1, np.int64)
rep[wid] = np.arange(len(H))
bnd = Eu[ec == 1]
# the open edge as dense points (1 mm), for the coat's fade
pa, pb = H[rep[bnd[:, 0]]], H[rep[bnd[:, 1]]]
nseg = np.maximum(np.ceil(np.linalg.norm(pb - pa, axis=1) / 0.001), 1).astype(int)
ring_pts = np.concatenate([pa[k] + np.linspace(0, 1, nseg[k] + 1)[:, None] * (pb[k] - pa[k]) for k in range(len(bnd))])
edge_in = np.abs(inside(ring_pts)) < 1e-3
print("C1 head piece: %d vertices, %d faces; open edge %.0f mm long, %.0f%% of it on the cut planes" % (
    len(H), len(Fh), 1000 * np.linalg.norm(pb - pa, axis=1).sum(), 100 * edge_in.mean()))
ring_tree = cKDTree(ring_pts)

# ---- the body fitted to the head (on the welded home mesh)
wb = weld_ids(P)
nW = wb.max() + 1
rep_b = np.zeros(nW, np.int64)
rep_b[wb] = np.arange(len(P))
Ew = np.unique(np.sort(np.concatenate([wb[F][:, [0, 1]], wb[F][:, [1, 2]], wb[F][:, [2, 0]]]), 1), axis=0)
Ew = Ew[Ew[:, 0] != Ew[:, 1]]
Adj = coo_matrix((np.ones(2 * len(Ew)), (np.r_[Ew[:, 0], Ew[:, 1]], np.r_[Ew[:, 1], Ew[:, 0]])), shape=(nW, nW)).tocsr()
deg = np.asarray(Adj.sum(1)).ravel()
Pw = P[rep_b]
s_w = inside(Pw)
near = np.linalg.norm(Pw - cen, axis=1) < 0.25
Nhp = vertex_normals(H, Fh)


def under_piece(X):
    """closest piece point, its normal and the height of X above it (along the normal)"""
    fi_, bc_, _ = closest_on_mesh(X, H, Fh)
    q_ = (bc_[:, :, None] * H[Fh[fi_]]).sum(1)
    n_ = (bc_[:, :, None] * Nhp[Fh[fi_]]).sum(1)
    n_ /= np.maximum(np.linalg.norm(n_, axis=1, keepdims=True), 1e-12)
    return q_, n_, np.sum((X - q_) * n_, 1)


# one smooth stretch of the home body near the head, solved as a whole (sparse least squares, FIT_ITERS rounds of
# closest points): as smooth as possible (the displacement's Laplacian) while vertices near the cut (BAND: 4 mm outside
# to INNER0 inside) sit DEPTH (deeper further in) beneath the piece along its normal and deep inside (past INNER1) go to
# their tucked place; no move at BLEND outside the cut. Moving each vertex on its own crumpled the neck (the second try).
U = np.where(near & (s_w > -BLEND))[0]
nU = len(U)
col = -np.ones(nW, np.int64)
col[U] = np.arange(nU)
Ln = (diags(np.ones(nW)) - diags(1 / np.maximum(deg, 1)) @ Adj).tocsr()
rows = np.unique(np.r_[U, Adj[U].indices])
LU = Ln[rows][:, U]
LtL = (LU.T @ LU).tocsr()
S3 = block_diag([LtL] * 3).tocsr()
bnd_u = U[(s_w[U] > -BAND_OUT) & (s_w[U] < INNER0)]
deep_u = U[s_w[U] > INNER1]
sb = s_w[bnd_u]
depth_b = DEPTH * sm(np.clip((sb + 0.004) / 0.004, 0, 1)) + SINK * np.maximum(sb, 0)
tuck = cen + TUCK * (Pw[deep_u] - cen)
D = np.zeros((nU, 3))
cb, cd = col[bnd_u], col[deep_u]
for it in range(FIT_ITERS):
    X = Pw[bnd_u] + D[cb]
    q, n, h = under_piece(X)
    r = np.sum(n * (q - Pw[bnd_u]), 1) - depth_b   # want n . D = r
    ri, ci, vv = [], [], []
    for a in range(3):
        for b in range(3):
            ri.append(a * nU + cb); ci.append(b * nU + cb); vv.append(FIT_A * n[:, a] * n[:, b])
    for a in range(3):
        ri.append(a * nU + cb); ci.append(a * nU + cb); vv.append(np.full(len(cb), FIT_G))
        ri.append(a * nU + cd); ci.append(a * nU + cd); vv.append(np.full(len(cd), FIT_B))
    M = S3 + coo_matrix((np.concatenate(vv), (np.concatenate(ri), np.concatenate(ci))), shape=(3 * nU, 3 * nU)).tocsr()
    rhs = np.zeros(3 * nU)
    tgt = q - depth_b[:, None] * n - Pw[bnd_u]
    for a in range(3):
        np.add.at(rhs, a * nU + cb, FIT_A * n[:, a] * r + FIT_G * tgt[:, a])
        np.add.at(rhs, a * nU + cd, FIT_B * (tuck[:, a] - Pw[deep_u, a]))
    D = spsolve(M.tocsc(), rhs).reshape(3, nU).T
    _, _, h = under_piece(Pw[bnd_u] + D[cb])
    err = h + depth_b
    print("fit round %d: near the cut, distance from the wanted depth median %.1f mm, p90 %.1f mm; above the piece %d of %d" % (
        it, 1000 * np.median(np.abs(err)), 1000 * np.percentile(np.abs(err), 90), int(((h > 0) & (sb > 0)).sum()), len(sb)))
Dw = np.zeros((nW, 3))
Dw[U] = D
# where the fit left a vertex near the cut above the piece (inside the cut) or less than a millimetre under it: pushed down
X = Pw[bnd_u] + Dw[bnd_u]
q, n, h = under_piece(X)
push = np.where(sb >= 0, np.maximum(h + 0.001, 0), 0)
Dw[bnd_u] -= push[:, None] * n
print("pushed under the piece after the fit: %d vertices (max %.1f mm)" % ((push > 0).sum(), 1000 * push.max()))
rough = lambda Y: np.linalg.norm(Y - (Adj @ Y) / np.maximum(deg, 1)[:, None], axis=1)  # noqa: E731
r0, r1 = rough(Pw)[U], rough(Pw + Dw)[U]
print("crinkle (vertex minus its neighbours' mean) near the head: median %.2f -> %.2f mm, p99 %.2f -> %.2f mm" % (
    1000 * np.median(r0), 1000 * np.median(r1), 1000 * np.percentile(r0, 99), 1000 * np.percentile(r1, 99)))
Ps = P + Dw[wb]
mv = np.linalg.norm(Ps - P, axis=1)
outside = np.where((s_w[wb] < 0) & (mv > 1e-6))[0]
print("body fitted: %d home vertices moved (%d outside the cut: median %.1f mm, max %.1f mm)" % (
    (mv > 1e-6).sum(), len(outside), 1000 * np.median(mv[outside]), 1000 * mv[outside].max()))

# ---- the piece's rim laid onto the body: within RIM of its open edge it bends down onto the fitted body (LIFT above
# it), so the edge sits on the neck with no step
Nb = vertex_normals(Ps, F)
de_v, _ = ring_tree.query(H)
a_r = sm(np.clip(1 - de_v / RIM, 0, 1))
hv = np.where(a_r > 0)[0]
bfi, bbc, bd = closest_on_mesh(H[hv], Ps, F)
bq = (bbc[:, :, None] * Ps[F[bfi]]).sum(1)
bn = (bbc[:, :, None] * Nb[F[bfi]]).sum(1)
bn /= np.maximum(np.linalg.norm(bn, axis=1, keepdims=True), 1e-12)
# under the chin, where the head's edge stands well clear of the neck (the jowls overhang it), it is left hanging:
# full laying below RIM_GAP, none above 1.5 RIM_GAP, smoothed along the piece so the edge does not wave; behind the
# head it is always laid down
behind = lambda X: (CUT - (X[:, 2] - zmin)) <= (CHIN - (X[:, 1] - nose[1]))  # noqa: E731  (CUT plane the nearer)
m = np.ones(len(H))
m[hv] = np.where(behind(H[hv]), 1.0, sm(np.clip((1.5 * RIM_GAP - bd) / (0.5 * RIM_GAP), 0, 1)))
Ep = np.unique(np.sort(np.concatenate([Fh[:, [0, 1]], Fh[:, [1, 2]], Fh[:, [2, 0]]]), 1), axis=0)
Ap = coo_matrix((np.ones(2 * len(Ep)), (np.r_[Ep[:, 0], Ep[:, 1]], np.r_[Ep[:, 1], Ep[:, 0]])), shape=(len(H),) * 2).tocsr()
Ap = Ap + coo_matrix((np.ones(len(wid)), (np.arange(len(wid)), rep[wid])), shape=(len(H),) * 2).tocsr()  # across uv seams
dp = np.asarray(Ap.sum(1)).ravel()
for _ in range(6):
    m = 0.5 * m + 0.5 * (Ap @ m) / np.maximum(dp, 1)
H = H.copy()
H[hv] += (a_r[hv] * m[hv])[:, None] * (bq + LIFT * bn - H[hv])
print("  left hanging (laying under half): %d of %d edge points" % (((m[hv] < 0.5) & (a_r[hv] > 0.99)).sum(), (a_r[hv] > 0.99).sum()))
print("rim: %d piece vertices laid onto the body, gap at the edge before: median %.1f mm, max %.1f mm" % (
    len(hv), 1000 * np.median(bd[a_r[hv] > 0.99]), 1000 * bd[a_r[hv] > 0.99].max()))
big = hv[(a_r[hv] > 0.99) & (bd > 0.006)]
print("  edge points over 6 mm from the body: %d; behind the nose %s mm, below it %s mm, side %s mm" % (
    len(big), np.round(1000 * (H[big, 2] - zmin)).astype(int).tolist()[:12], np.round(1000 * (H[big, 1] - nose[1])).astype(int).tolist()[:12],
    np.round(1000 * (H[big, 0] - nose[0])).astype(int).tolist()[:12]))
pa, pb = H[rep[bnd[:, 0]]], H[rep[bnd[:, 1]]]
ring_pts = np.concatenate([pa[k] + np.linspace(0, 1, nseg[k] + 1)[:, None] * (pb[k] - pa[k]) for k in range(len(bnd))])
ring_tree = cKDTree(ring_pts)
# does any body vertex stand proud of the (final) piece where the piece covers it?
Nhp = vertex_normals(H, Fh)
bi_ = np.where(mv > 1e-6)[0]
pq, _, proud = under_piece(Ps[bi_])
over = inside(pq) > 0.002
print("body vertices under the piece (more than 2 mm in from its edge): %d; standing proud of it: %d (max %.2f mm)" % (
    over.sum(), (over & (proud > 0)).sum(), 1000 * max(proud[over].max(), 0) if over.any() else 0))
pr = bi_[over & (proud > 0)]
if len(pr):
    print("  proud ones: inside the cut by %s mm; behind the nose %s mm, below it %s mm, side %s mm; moved %s mm" % (
        np.round(1000 * inside(P[pr])).astype(int).tolist()[:12], np.round(1000 * (P[pr, 2] - zmin)).astype(int).tolist()[:12],
        np.round(1000 * (P[pr, 1] - nose[1])).astype(int).tolist()[:12], np.round(1000 * (P[pr, 0] - nose[0])).astype(int).tolist()[:12],
        np.round(1000 * mv[pr]).astype(int).tolist()[:12]))

# ---- skinning of the piece from the nearest 4 fitted body vertices (inverse distance)
idx, sw = A["skin_idx"].astype(np.int64), A["skin_w"].astype(np.float64)
nb = A["skin_idx"].max() + 1
d, nn = cKDTree(Ps).query(H, k=4)
wt = 1 / np.maximum(d, 1e-4)
wt /= wt.sum(1, keepdims=True)
Wfull = np.zeros((len(H), max(int(nb), 25)))
for k in range(4):
    np.add.at(Wfull, (np.repeat(np.arange(len(H)), 4), idx[nn[:, k]].ravel()), (sw[nn[:, k]] * wt[:, k:k + 1]).ravel())
top = np.argsort(-Wfull, 1)[:, :4]
hw = np.take_along_axis(Wfull, top, 1)
hw /= hw.sum(1, keepdims=True)

# ---- the piece's coat: C1's photo coat, fading into the body's coat at the open edge
S = Tc.shape[0]
tpos, trc = rasterise(H, Fh, UVh, S)
photo = Tc[trc[:, 0], trc[:, 1]].copy()
if COVER:  # as the whole-cat coat (home_coat.py FILL): spots C1's video never saw, or much darker than the fur within
    # 5 cm (outside the front 3 cm of the face), take a smooth fill from the reliable fur around them
    cov = np.asarray(Image.open(COVER).convert("L"), dtype=np.float32) / 255.0
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
    face_front = tpos[:, 2] - zmin < 0.03
    dark[face_front] = 0
    rel = seen * (1 - dark)
    photo = rel[:, None] * photo + (1 - rel[:, None]) * voxel_fill(photo, tpos, rel)
    print("head coat fill: %.0f%% of its texels unseen by C1's video, %.0f%% much darker than the fur round them; "
          "lightness under 35 outside the face %.1f%% -> %.1f%%" % (100 * (seen < 0.5).mean(), 100 * (dark > 0.5).mean(),
          100 * ((Lp < 35) & ~face_front).mean(), 100 * ((lightness(photo) < 35) & ~face_front).mean()))
de, _ = ring_tree.query(tpos)
f = 1 - sm(np.clip(de / FADE, 0, 1))
sel = np.where(f > 1e-3)[0]
bfi, bbc, _ = closest_on_mesh(tpos[sel], Ps, F)
cb = sample(T, (bbc[:, :, None] * UV[F[bfi]]).sum(1))
Tn = Tc.copy()
Tn[trc[:, 0], trc[:, 1]] = photo
r, c = trc[sel, 0], trc[sel, 1]
Tn[r, c] = f[sel, None] * cb + (1 - f[sel, None]) * photo[sel]
r, c = trc[:, 0], trc[:, 1]
# carry the change a few texels into the atlas gutter (texels of no C1 triangle), for filtering
_, crc = rasterise(Xh, Fc, UVc.astype(np.float64), S)
covered = np.zeros((S, S), bool)
covered[crc[:, 0], crc[:, 1]] = True
changed = np.zeros((S, S), bool)
changed[r, c] = True
for _ in range(4):
    acc, cnt = np.zeros_like(Tn), np.zeros((S, S))
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        acc += np.roll(np.roll(Tn * changed[..., None], dr, 0), dc, 1)
        cnt += np.roll(np.roll(changed, dr, 0), dc, 1)
    grow = ~covered & ~changed & (cnt > 0)
    Tn[grow] = acc[grow] / cnt[grow][:, None]
    changed |= grow
dl = np.abs(Tn[trc[sel, 0], trc[sel, 1]] - photo[sel]).max(1)
print("coat: %d piece texels, %d faded towards the body's coat (median change %.0f/255 where faded more than half)" % (
    len(tpos), len(sel), 255 * np.median(dl[f[sel] > 0.5]) if (f[sel] > 0.5).any() else 0))
Image.fromarray((Tn * 255).round().clip(0, 255).astype(np.uint8)).save(os.path.join(out, "a2s_graft_tex.jpg"), quality=90)
np.savez(os.path.join(out, "graft.npz"), H=H.astype(np.float32), Fh=Fh.astype(np.uint32), UVh=UVh.astype(np.float32),
         hidx=top.astype(np.uint8), hw=hw.astype(np.float32), Ps=Ps.astype(np.float32))

# ---- the page's file: body offsets (rest pose) and the head piece
mvi = np.where(mv > 1e-6)[0].astype(np.uint32)
blobs = [("body_idx", mvi), ("body_off", (Ps - P)[mvi].astype(np.float32)), ("pos", H.astype(np.float32)),
         ("faces", Fh.astype(np.uint32)), ("uv", UVh.astype(np.float32)), ("skin_idx", top.astype(np.uint8)),
         ("skin_w", hw.astype(np.float32))]
meta, nbytes = pack(os.path.join(out, "a2s_graft.bin"), blobs)
json.dump({"blobs": meta, "bytes": nbytes, "vertices": len(H), "texture": "a2s_graft_tex.jpg",
           "units": "metres, the cat's own (root) frame; body_off is added to a2s_cat pos",
           "from": "head_graft.py: the Body tab cat's head (cut behind the ears and below the chin) on the home body; "
                   "CUT %.3f CHIN %.3f BLEND %.3f DEPTH %.4f FADE %.3f RIM %.3f RIM_GAP %.3f FIT_A %.2f, head coat filled: %s" % (
                       CUT, CHIN, BLEND, DEPTH, FADE, RIM, RIM_GAP, FIT_A, "yes" if COVER else "no"),
           "chunks": 1}, open(os.path.join(out, "a2s_graft.json"), "w"))
print("page file: %d body vertices move, head piece %d vertices, %d bytes" % (len(mvi), len(H), nbytes), flush=True)


if os.environ.get("FIT_ONLY"):
    sys.exit(0)


# ---- renders
def turn(X, deg):
    a = np.radians(deg)
    return X @ np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]]).T


def both(Xb, Xhd, view, crop_c, crop_r, tex=Tn):
    """the body and the head piece in one render: one mesh, the two textures side by side"""
    Tw = np.concatenate([T, tex], 1)
    UVb = UV.copy()
    UVb[:, 0] *= 0.5
    UVg = UVh.copy()
    UVg[:, 0] = 0.5 + 0.5 * UVg[:, 0]
    X = np.concatenate([Xb, Xhd])
    Fx = np.concatenate([F, Fh + len(Xb)])
    UVx = np.concatenate([UVb, UVg])
    crop = None
    if crop_c is not None:
        crop = np.concatenate([np.linalg.norm(Xb - crop_c, axis=1) < crop_r, np.linalg.norm(Xhd - crop_c, axis=1) < crop_r])
    return render(X, Fx, UVx, Tw, view, crop=crop)


def save(row, name):
    Image.fromarray((np.concatenate(row, 1) * 255).clip(0, 255).astype(np.uint8)).save(os.path.join(out, name))


c0 = P[np.argsort(P[:, 2])[:20]].mean(0)
for view in ("front", "side", "other_side", "three_quarter"):
    save([render(P, F, UV, T, view, crop=np.linalg.norm(P - c0, axis=1) < 0.13), both(Ps, H, view, c0, 0.13)],
         "graft_%s.png" % view)
# the seam close up: behind the head, from behind and above at 140 and 220 degrees, and from below
cs = cen + np.array([0, 0, 0.03])
row = []
for deg in (140, 220):
    row.append(both(turn(Ps - cs, deg), turn(H - cs, deg), "front", np.zeros(3), 0.08))
save(row, "graft_seam_back.png")
save([both(Ps, H, "side", cs, 0.08), both(Ps, H, "other_side", cs, 0.08)], "graft_seam_sides.png")
# under the chin: from below (looking up), and the plain coat (no fade) for comparison
lowv = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], float)  # y (down) becomes depth: the camera looks up from below
save([both((Ps - c0) @ lowv.T, (H - c0) @ lowv.T, "front", np.zeros(3), 0.11),
      both((Ps - c0) @ lowv.T, (H - c0) @ lowv.T, "front", np.zeros(3), 0.11, tex=Tc)], "graft_chin_below.png")
save([render(P, F, UV, T, "side"), both(Ps, H, "side", None, None)], "graft_side_whole.png")
bones = A["bones"].astype(np.float64).reshape(len(A["bones"]), A["bones"].shape[1], 3, 4)
for fr in (500, 3000, 6000, 9000):
    Xo, Xs, Xg = lbs(P, idx, sw, bones[fr]), lbs(Ps, idx, sw, bones[fr]), lbs(H, top, hw, bones[fr])
    c = Xo[np.argsort(P[:, 2])[:20]].mean(0)
    row = [render(Xo, F, UV, T, v, crop=np.linalg.norm(Xo - c, axis=1) < 0.14) for v in ("side", "front")]
    row += [both(Xs, Xg, v, c, 0.14) for v in ("side", "front")]
    save(row, "graft_posed_%05d.png" % fr)
print("renders done", flush=True)
