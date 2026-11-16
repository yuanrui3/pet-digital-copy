"""Second cat at home (work/pets/PROTOCOL_CAT.md, "Second cat at home"): fit the grey and white cat's rest mesh onto
Agent-to-Sim's Pikachu, transfer Pikachu's skinning to it, and check how it moves under Pikachu's bone motion.
CPU only (numpy, scipy, matplotlib).

usage: python home2_transfer.py A2S_DIR UV_NPZ CAT2_VIEWER_DIR PATHS_DIR OUT_DIR
  A2S_DIR: a2s_cat.bin/.json (Pikachu's rest mesh, skinning and recorded bone motion, from a2s_cat_export.py)
  UV_NPZ: the grey cat's UV-mapped rest mesh (cat_uv_bake.py); CAT2_VIEWER_DIR: its viewer_data.bin/viewer_meta.json
  PATHS_DIR: paths.bin/.json (the recorded root heading per frame, same order as the bone motion; the room's y points down)
Writes OUT_DIR/a2s_cat2.bin + a2s_cat2.json (the page's format, no bones: the page uses Pikachu's), home2_check.json
and home2_check.png.
"""
import json
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

K_SKIN, SMOOTH_ITERS, ICP_ITERS, CHECK_FRAMES = 8, 5, 60, 40


def read_blobs(path, meta):
    buf = open(path, "rb").read()
    return {b["name"]: np.frombuffer(buf, dtype=b["dtype"], count=int(np.prod(b["shape"])), offset=b["offset"])
            .reshape(b["shape"]) for b in meta["blobs"]}


def pack(path, blobs):
    meta, off = [], 0
    with open(path, "wb") as fh:
        for name, arr in blobs:
            b = np.ascontiguousarray(arr).tobytes()
            meta.append({"name": name, "dtype": str(arr.dtype), "shape": list(arr.shape), "offset": off, "bytes": len(b)})
            fh.write(b + b"\0" * ((-len(b)) % 4))
            off += len(b) + (-len(b)) % 4
    return meta, off


def umeyama(src, dst):
    """similarity (s, R, t) with dst ~ s R src + t"""
    ms, md = src.mean(0), dst.mean(0)
    a, b = src - ms, dst - md
    U, S, Vt = np.linalg.svd(b.T @ a / len(src))
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ D @ Vt
    s = (S * np.diag(D)).sum() / (a ** 2).sum(1).mean()
    return s, R, md - s * R @ ms


def icp(Q, P, s, R, t, tree_p):
    for _ in range(ICP_ITERS):
        Qa = s * Q @ R.T + t
        _, i_qp = tree_p.query(Qa)  # each grey-cat point -> nearest Pikachu point
        _, i_pq = cKDTree(Qa).query(P)  # each Pikachu point -> nearest grey-cat point
        src = np.concatenate([Q, Q[i_pq]])
        dst = np.concatenate([P[i_qp], P])
        s, R, t = umeyama(src, dst)
    Qa = s * Q @ R.T + t
    d1, _ = tree_p.query(Qa)
    d2, _ = cKDTree(Qa).query(P)
    return s, R, t, float((d1.mean() + d2.mean()) / 2), d1, d2


def lbs(rest, idx, w, bones):
    """rest (V,3), idx/w (V,4), bones (B,3,4) -> posed (V,3)"""
    T = bones[idx.astype(np.int64)]  # (V,4,3,4)
    p = np.einsum("vkij,vj->vki", T[..., :3], rest) + T[..., 3]
    return (p * w[..., None]).sum(1)


def edges_of(faces):
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    return np.unique(np.sort(e, 1), axis=0)


def stretch(rest, posed, e):
    l0 = np.linalg.norm(rest[e[:, 0]] - rest[e[:, 1]], axis=1)
    l1 = np.linalg.norm(posed[e[:, 0]] - posed[e[:, 1]], axis=1)
    ok = l0 > 1e-6
    return np.abs(np.log(np.maximum(l1[ok], 1e-9) / l0[ok]))


def main():
    a2s_dir, uv_npz, v2_dir, paths_dir, out = sys.argv[1:6]
    pm = json.load(open(os.path.join(paths_dir, "paths.json")))
    heading = read_blobs(os.path.join(paths_dir, "paths.bin"), pm)["heading"].astype(np.float64).reshape(-1, 3, 3)
    os.makedirs(out, exist_ok=True)
    am = json.load(open(os.path.join(a2s_dir, "a2s_cat.json")))
    A = read_blobs(os.path.join(a2s_dir, "a2s_cat.bin"), am)
    P = A["pos"].astype(np.float64)
    B = am["bones"]
    bones_all = A["bones"].astype(np.float32).reshape(-1, B, 3, 4)
    z = np.load(uv_npz)
    Q, faces2, uv2 = z["pos"].astype(np.float64), z["faces"].astype(np.int64), z["uv"].astype(np.float32)
    vm = json.load(open(os.path.join(v2_dir, "viewer_meta.json")))
    V2 = read_blobs(os.path.join(v2_dir, "viewer_data.bin"), vm)
    assert len(Q) == vm["vertices"], (len(Q), vm["vertices"])
    col2 = np.asarray(V2["col_fixed"], dtype=np.uint8)
    print("Pikachu (Agent-to-Sim): %d vertices; grey cat: %d vertices, %d faces" % (len(P), len(Q), len(faces2)))

    # ---- fit: one scale, rotation and shift, from the 4 proper principal-axis starts
    tree_p = cKDTree(P)
    mp, mq = P.mean(0), Q.mean(0)
    ep = np.linalg.eigh(np.cov((P - mp).T))[1][:, ::-1]
    eq = np.linalg.eigh(np.cov((Q - mq).T))[1][:, ::-1]
    s0 = np.sqrt(((P - mp) ** 2).sum(1).mean() / ((Q - mq) ** 2).sum(1).mean())
    size = float(np.linalg.norm(P.max(0) - P.min(0)))
    fits = []
    for sg in ([1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]):
        R0 = ep @ np.diag(sg) @ eq.T
        if np.linalg.det(R0) < 0:
            R0 = ep @ np.diag(np.array(sg) * [1, 1, -1]) @ eq.T
        fit = icp(Q, P, s0, R0, mp - s0 * R0 @ mq, tree_p)
        fits.append(fit)
        print("start %s: fit distance %.1f mm (scale %.4f)" % (sg, 1000 * fit[3], fit[0]))
    s, R, t, fd, d1, d2 = min(fits, key=lambda f: f[3])
    Qa = s * Q @ R.T + t
    print("fit: scale %.4f m per model unit, mean distance %.1f mm (%.1f%% of the cat's size), 95th pct %.1f / %.1f mm"
          % (s, 1000 * fd, 100 * fd / size, 1000 * np.percentile(d1, 95), 1000 * np.percentile(d2, 95)))

    # ---- optional non-rigid fit (NONRIGID=1): bend the grey cat's rest mesh smoothly onto Pikachu's rest pose; the bent
    # mesh is then the grey cat's rest mesh, so each part sits where Pikachu's matching part is and follows its bones
    e2 = edges_of(faces2)
    nonrigid = os.environ.get("NONRIGID", "0") == "1"
    Q_rigid = Qa.copy()
    if nonrigid:
        from scipy.sparse import coo_matrix, diags
        from scipy.sparse.linalg import factorized
        n = len(Qa)
        Lw = coo_matrix((np.ones(2 * len(e2)), (np.r_[e2[:, 0], e2[:, 1]], np.r_[e2[:, 1], e2[:, 0]])), shape=(n, n)).tocsr()
        L = diags(np.asarray(Lw.sum(1)).ravel()) - Lw  # graph Laplacian
        X = Qa.copy()
        for lam in [float(x) for x in os.environ.get("NR_LAMS", "30 10 3 1").split()]:
            solve = factorized((diags(np.full(n, 1.0)) + lam * L).tocsc())
            for _ in range(8):
                _, fi = tree_p.query(X)  # each grey vertex -> nearest Pikachu vertex
                tgt = P[fi].copy()
                cnt = np.ones(n)
                _, ri = cKDTree(X).query(P)  # each Pikachu vertex pulls its nearest grey vertex
                np.add.at(tgt, ri, P)
                np.add.at(cnt, ri, 1)
                tgt /= cnt[:, None]
                U = np.stack([solve(tgt[:, k] - Qa[:, k]) for k in range(3)], 1)  # total displacement, kept smooth
                X = Qa + U
            dn = tree_p.query(X)[0]
            print("non-rigid, smoothness %g: mean distance %.1f mm" % (lam, 1000 * dn.mean()))
        el0 = np.linalg.norm(Qa[e2[:, 0]] - Qa[e2[:, 1]], axis=1)
        el1 = np.linalg.norm(X[e2[:, 0]] - X[e2[:, 1]], axis=1)
        ch = np.abs(np.log(np.maximum(el1, 1e-9) / np.maximum(el0, 1e-9)))
        print("non-rigid: the grey cat's own edges changed length by median %.1f%%, 95th pct %.1f%%; moved median %.1f mm"
              % (100 * (np.exp(np.median(ch)) - 1), 100 * (np.exp(np.percentile(ch, 95)) - 1),
                 1000 * np.median(np.linalg.norm(X - Qa, axis=1))))
        Qa = X
        fd = float((tree_p.query(Qa)[0].mean() + cKDTree(Qa).query(P)[0].mean()) / 2)

    # ---- skinning transfer: 8 nearest Pikachu vertices, then smoothed over the grey cat's mesh
    d, i = tree_p.query(Qa, k=K_SKIN)
    wk = 1.0 / np.maximum(d, 1e-4)
    W = np.zeros((len(Qa), B))
    for k in range(K_SKIN):
        np.add.at(W, (np.arange(len(Qa))[:, None], A["skin_idx"][i[:, k]].astype(np.int64)),
                  (wk[:, k:k + 1] * A["skin_w"][i[:, k]]))
    W /= W.sum(1, keepdims=True)
    nb_sum = np.zeros_like(W)
    deg = np.bincount(e2.ravel(), minlength=len(Qa))[:, None].astype(np.float64)
    for _ in range(SMOOTH_ITERS):
        nb_sum[:] = 0
        np.add.at(nb_sum, e2[:, 0], W[e2[:, 1]])
        np.add.at(nb_sum, e2[:, 1], W[e2[:, 0]])
        W = 0.5 * W + 0.5 * nb_sum / np.maximum(deg, 1)
    i4 = np.argsort(-W, 1)[:, :4]
    w4 = np.take_along_axis(W, i4, 1)
    cover = float(w4.sum(1).mean())
    w4 = (w4 / w4.sum(1, keepdims=True)).astype(np.float32)
    i4 = i4.astype(np.uint8)

    # ---- check on 40 recorded frames: edge stretch, and where the lowest point sits against Pikachu's
    frames = np.linspace(0, len(bones_all) - 1, CHECK_FRAMES).round().astype(int)
    e1 = edges_of(A["faces"].astype(np.int64))
    st1, st2, low_gap = [], [], []
    for f in frames:
        bo = bones_all[f]
        p1 = lbs(P, A["skin_idx"], A["skin_w"], bo)
        p2 = lbs(Qa, i4, w4, bo)
        st1.append(stretch(P, p1, e1))
        st2.append(stretch(Qa, p2, e2))
        down = heading[f][1]  # the room's y (down) in the cat's own frame
        low_gap.append((p2 @ down).max() - (p1 @ down).max())  # > 0: the grey cat reaches lower (sinks)
    st1, st2 = np.concatenate(st1), np.concatenate(st2)
    q1, q2 = float(np.percentile(st1, 99)), float(np.percentile(st2, 99))
    ratio = q2 / max(q1, 1e-9)
    readout = "moves cleanly" if ratio <= 1.5 else ("some stretching" if ratio <= 3 else "tears")
    check = {
        "nonrigid": nonrigid, "fit_scale_m_per_unit": float(s), "fit_mean_distance_mm": 1000 * fd, "fit_share_of_size": fd / size,
        "skin_top4_cover": cover, "frames_checked": len(frames),
        "stretch_p99_pikachu": q1, "stretch_p99_grey": q2, "ratio": ratio, "readout": readout,
        "stretch_median_pikachu": float(np.median(st1)), "stretch_median_grey": float(np.median(st2)),
        "share_edges_over_30pct_pikachu": float((st1 > np.log(1.3)).mean()),
        "share_edges_over_30pct_grey": float((st2 > np.log(1.3)).mean()),
        "lowest_point_vs_pikachu_mm_median": 1000 * float(np.median(low_gap)),
        "lowest_point_vs_pikachu_mm_p10_p90": [1000 * float(np.percentile(low_gap, 10)), 1000 * float(np.percentile(low_gap, 90))],
        "grey_size_m": float(np.linalg.norm(Qa.max(0) - Qa.min(0))), "pikachu_size_m": size,
    }
    json.dump(check, open(os.path.join(out, "home2_check.json"), "w"), indent=1)
    print(json.dumps(check, indent=1))

    meta, nbytes = pack(os.path.join(out, "a2s_cat2.bin"), [
        ("faces", faces2.astype(np.uint32)), ("pos", Qa.astype(np.float32)), ("col", col2),
        ("skin_idx", i4), ("skin_w", w4), ("uv", uv2)])
    json.dump({"blobs": meta, "bytes": nbytes, "vertices": len(Qa), "bones": B, "frames": 0, "texture": "cat2_tex_photo.jpg",
               "units": "metres, Pikachu's (Agent-to-Sim's) own frame",
               "from": "the grey and white cat's rest mesh fitted onto Pikachu's; Pikachu's skinning transferred"},
              open(os.path.join(out, "a2s_cat2.json"), "w"))
    print("a2s_cat2: %d bytes" % nbytes)

    # ---- picture: four frames from the side and from above, Pikachu (orange) and the grey cat (blue)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 5, figsize=(22, 8))
    show = [None] + list(frames[[5, 15, 25, 35]])
    for c, f in enumerate(show):
        if f is None:
            p1, p2, title = P, Qa, "rest"
        else:
            p1 = lbs(P, A["skin_idx"], A["skin_w"], bones_all[f])
            p2 = lbs(Qa, i4, w4, bones_all[f])
            title = "recorded frame %d" % f
        for r, (a, b) in enumerate(((0, 1), (0, 2))):
            ax = axes[r, c]
            ax.scatter(p1[::3, a], p1[::3, b], s=0.3, c="tab:orange", alpha=0.4)
            ax.scatter(p2[::3, a], p2[::3, b], s=0.3, c="tab:blue", alpha=0.4)
            ax.set_aspect("equal")
            if r == 0:
                ax.invert_yaxis()
                ax.set_title(title)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "home2_check.png"), dpi=80)


if __name__ == "__main__":
    main()
