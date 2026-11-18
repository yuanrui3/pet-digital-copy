"""Second cat at home, retarget (work/pets/PROTOCOL_CAT.md, "Second cat at home"): carry Pikachu's (Agent-to-Sim's)
motion over onto the grey and white cat's own body and skinning.

Both skeletons have the same 25 joints, tree and order (both come from the same quad URDF), but both were learnt
freely, so a bone does not move the same body part in the two cats (the grey cat's "L hand" moves its right front leg).
Copying motion bone by bone therefore fails (MODE=dir: turn each grey bone onto Pikachu's bone direction; MODE=none:
the same turn from rest; both tried first, both recorded). Used, MODE=region:
  - REBASE=auto: the grey cat's canonical rest pose is bent sideways with the paws tucked together, so it is first
    posed as in the frame of its own video that best fits Pikachu's rest pose and stands on four clear paws;
  - that pose is lined up with Pikachu's rest mesh (one scale, rotation, shift) and bent smoothly onto it, paw tips,
    nose and tail tip pinned (only to find which part of Pikachu each grey vertex corresponds to);
  - each grey bone gets a mix of Pikachu's bones (their skinning weights at its vertices' corresponding points) and a
    centre; per frame it moves rigidly with that blend (blended quaternions; translation: where the blend carries the
    centre). The page does this in the browser from Pikachu's bone motion (index.html, poseGrey), and also raises or
    lowers the grey cat each frame so its lowest point sits where Pikachu's does.
CPU only (numpy, scipy, matplotlib).

usage: python home2_retarget.py PAGE_DIR UV_NPZ SKEL_NPZ PATHS_DIR OUT_DIR
  PAGE_DIR: the page's a2s_cat.bin/.json and a2s_walks.bin/.json (Pikachu's rest mesh, skinning, bone motion; metres)
  UV_NPZ: the grey cat's UV-mapped rest mesh; SKEL_NPZ: its skeleton, skinning and own motion (home2_skel.py)
  PATHS_DIR: paths.bin/.json (recorded root heading per frame, same order as the bone motion; the room's y points down)
Writes OUT_DIR/a2s_cat2.bin + .json (NAME and TEX change the file and texture names; the whole Body tab cat at home
uses a2s_cat3) (the grey cat's rebased rest mesh in Pikachu's frame, its own skinning, and the
constants the page needs), home2_retarget.json and home2_retarget.png.
"""
import json
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from home2_transfer import edges_of, icp, lbs, pack, read_blobs, stretch  # noqa: E402

NAMES = ["spine 1", "spine 2", "spine 3", "head", "L shoulder", "L elbow", "L wrist", "L hand", "R shoulder",
         "R elbow", "R wrist", "R hand", "tail 1", "tail 2", "tail 3", "tail 4", "tail 5", "L hip", "L knee",
         "L ankle", "L foot", "R hip", "R knee", "R ankle", "R foot"]
LEGS = {"L front": [4, 5, 6, 7], "R front": [8, 9, 10, 11], "L hind": [17, 18, 19, 20], "R hind": [21, 22, 23, 24]}
CHECK_FRAMES = 40


def skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def rot_between(a, b):
    """smallest rotation taking direction a onto direction b"""
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v, c = np.cross(a, b), float(a @ b)
    if c < -0.999999:
        p = np.cross(a, [1.0, 0, 0] if abs(a[0]) < 0.9 else [0, 1.0, 0])
        p /= np.linalg.norm(p)
        return 2 * np.outer(p, p) - np.eye(3)
    k = skew(v)
    return np.eye(3) + k + k @ k / (1 + c)


def kabsch(src, dst):
    """rotation R with dst[k] ~ R src[k]"""
    U, _, Vt = np.linalg.svd(dst.T @ src)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))])
    return U @ D @ Vt


def joints_from_bones(T, parents):
    """each joint is the point that moves with both its bone and its parent bone (the root is fixed): least squares
    over frames, in the rest space. T (F,B,3,4) rest -> frame."""
    F, B = T.shape[:2]
    out, res = np.zeros((B, 3)), np.zeros(B)
    for i in range(B):
        Tp = T[:, parents[i]] if parents[i] >= 0 else np.broadcast_to(np.eye(3, 4), (F, 3, 4))
        M, d = T[:, i, :, :3] - Tp[..., :3], T[:, i, :, 3] - Tp[..., 3]
        x = -np.linalg.solve(np.einsum("fji,fjk->ik", M, M), np.einsum("fji,fj->i", M, d))
        out[i] = x
        res[i] = np.median(np.linalg.norm(M @ x + d, axis=1))
    return out, res


def children_of(parents):
    ch = [[] for _ in parents]
    for i, p in enumerate(parents):
        if p >= 0:
            ch[p].append(i)
    return ch


def rest_alignment(p, q, parents):
    """per bone: the fixed rotation that turns the grey bone's rest direction(s) onto Pikachu's"""
    ch = children_of(parents)
    A = np.zeros((len(p), 3, 3))
    for i in range(len(p)):
        if len(ch[i]) >= 2:
            A[i] = kabsch(np.array([(q[c] - q[i]) / np.linalg.norm(q[c] - q[i]) for c in ch[i]]),
                          np.array([(p[c] - p[i]) / np.linalg.norm(p[c] - p[i]) for c in ch[i]]))
        elif len(ch[i]) == 1:
            c = ch[i][0]
            A[i] = rot_between(q[c] - q[i], p[c] - p[i])
        else:
            A[i] = A[parents[i]]  # a tip (head, paws, tail end): as its parent
    return A


def retarget(T, A, q, parents):
    """Pikachu's bone motion T (F,B,3,4) -> the grey cat's (F,B,3,4), rest -> frame; parents come before children.
    A joint on the root (spine 1, tail 1, the hips) is carried by Pikachu's matching bone, so the body turns about
    Pikachu's pivots; every other joint follows from its parent with the grey cat's own bone lengths."""
    F, B = T.shape[:2]
    G = np.zeros((F, B, 3, 4))
    pos = np.zeros((F, B, 3))
    for i in range(B):
        G[:, i, :, :3] = T[:, i, :, :3] @ A[i]
        p = parents[i]
        pos[:, i] = T[:, i, :, :3] @ q[i] + T[:, i, :, 3] if p < 0 else pos[:, p] + G[:, p, :, :3] @ (q[i] - q[p])
        G[:, i, :, 3] = pos[:, i] - G[:, i, :, :3] @ q[i]
    return G


def quat_from_mat(R):
    """(...,3,3) rotations -> (...,4) unit quaternions (w, x, y, z)"""
    m = R
    c = np.stack([
        np.stack([1 + m[..., 0, 0] + m[..., 1, 1] + m[..., 2, 2], m[..., 2, 1] - m[..., 1, 2], m[..., 0, 2] - m[..., 2, 0], m[..., 1, 0] - m[..., 0, 1]], -1),
        np.stack([m[..., 2, 1] - m[..., 1, 2], 1 + m[..., 0, 0] - m[..., 1, 1] - m[..., 2, 2], m[..., 0, 1] + m[..., 1, 0], m[..., 0, 2] + m[..., 2, 0]], -1),
        np.stack([m[..., 0, 2] - m[..., 2, 0], m[..., 0, 1] + m[..., 1, 0], 1 - m[..., 0, 0] + m[..., 1, 1] - m[..., 2, 2], m[..., 1, 2] + m[..., 2, 1]], -1),
        np.stack([m[..., 1, 0] - m[..., 0, 1], m[..., 0, 2] + m[..., 2, 0], m[..., 1, 2] + m[..., 2, 1], 1 - m[..., 0, 0] - m[..., 1, 1] + m[..., 2, 2]], -1)], -2)
    k = np.argmax(np.stack([m[..., 0, 0] + m[..., 1, 1] + m[..., 2, 2], m[..., 0, 0], m[..., 1, 1], m[..., 2, 2]], -1), -1)
    q = np.take_along_axis(c, k[..., None, None], -2)[..., 0, :]
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def mat_from_quat(q):
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack([np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
                     np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
                     np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1)], -2)


def paw_tips(X, e):
    """the four paw tips in a pose (y down): pieces of the mesh's lowest part, tips (mean of each piece's 20 lowest
    points) closer than 4 cm merged; the lowest part is widened until there are 4. None if no clear 4."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    n = len(X)
    for pct in (90, 88, 85, 82, 78, 74):
        low = X[:, 1] > np.percentile(X[:, 1], pct)
        ee = e[low[e[:, 0]] & low[e[:, 1]]]
        _, lab = connected_components(coo_matrix((np.ones(len(ee)), (ee[:, 0], ee[:, 1])), shape=(n, n)), directed=False)
        labs, cnt = np.unique(lab[low], return_counts=True)
        tips = []
        for c in labs[np.argsort(-cnt)]:
            v = np.where((lab == c) & low)[0]
            if len(v) < 20:
                break
            t = X[v[np.argsort(-X[v, 1])[:20]]].mean(0)
            if all(np.linalg.norm(t - u) > 0.04 for u in tips):
                tips.append(t)
        if len(tips) >= 4:
            tips = sorted(tips[:4], key=lambda t: t[2])
            if tips[2][2] - tips[1][2] < 0.05:  # front and hind pairs must be clearly apart
                return None
            return tips
    return None


def landmarks(X, faces):
    """rest pose, Pikachu's frame (y down, z towards the tail): the four paw tips (front pair by z, sides by x), the
    nose (front-most point) and the tail tip (highest point of the rear 20%)"""
    tips = paw_tips(X, edges_of(faces))
    assert tips is not None, "no clear four paws"
    out = {}
    for pair, name in ((tips[:2], "front"), (tips[2:], "hind")):
        a, b = sorted(pair, key=lambda t: t[0])
        out["paw -x " + name], out["paw +x " + name] = a, b
    out["nose"] = X[np.argsort(X[:, 2])[:20]].mean(0)
    rear = X[X[:, 2] > np.percentile(X[:, 2], 80)]
    out["tail tip"] = rear[np.argsort(rear[:, 1])[:20]].mean(0)
    return out


def nonrigid(Qa, faces, P, faces_p, tree_p, lams, lm_w=20.0, lm_r=0.025):
    """bend the grey cat's rest mesh smoothly onto Pikachu's, its paw tips, nose and tail tip pinned onto Pikachu's
    (used only to find which part of Pikachu each grey vertex corresponds to; the bent mesh is never shown)"""
    from scipy.sparse import coo_matrix, diags
    from scipy.sparse.linalg import factorized
    e = edges_of(faces)
    n = len(Qa)
    Lw = coo_matrix((np.ones(2 * len(e)), (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])), shape=(n, n)).tocsr()
    L = diags(np.asarray(Lw.sum(1)).ravel()) - Lw
    lq, lp = landmarks(Qa, faces), landmarks(P, faces_p)
    lp = {k: lp[k] for k in lq}
    wd = np.ones(n)
    pin = np.full(n, -1)
    for k, name in enumerate(lq):
        near = np.linalg.norm(Qa - lq[name], axis=1) < lm_r
        wd[near], pin[near] = lm_w, k
        print("  landmark %-12s grey %s -> Pikachu %s (%d vertices pinned)" % (
            name, np.round(lq[name], 3), np.round(lp[name], 3), near.sum()), flush=True)
    shift = np.array([lp[name] - lq[name] for name in lq])
    X = Qa.copy()
    for lam in lams:
        solve = factorized((diags(wd) + lam * L).tocsc())
        for _ in range(8):
            _, fi = tree_p.query(X)
            tgt, cnt = P[fi].copy(), np.ones(n)
            _, ri = cKDTree(X).query(P)
            np.add.at(tgt, ri, P)
            np.add.at(cnt, ri, 1)
            tgt /= cnt[:, None]
            tgt[pin >= 0] = Qa[pin >= 0] + shift[pin[pin >= 0]]
            X = Qa + np.stack([solve(wd * (tgt[:, k] - Qa[:, k])) for k in range(3)], 1)
        print("  correspondence bend, smoothness %g: mean distance %.1f mm" % (lam, 1000 * tree_p.query(X)[0].mean()), flush=True)
    return X


def region_mix(Qa, X, i4g, w4g, P, i4p, w4p, B, top=6, k=4):
    """for each grey bone: which of Pikachu's bones move the flesh its own vertices correspond to (a mix over
    Pikachu's bones, top 6 kept), and the centre of its vertices"""
    d, nn = cKDTree(P).query(X, k=k)
    wv = 1.0 / np.maximum(d, 1e-4)
    wv /= wv.sum(1, keepdims=True)
    n = np.arange(len(X))[:, None]
    Wp = np.zeros((len(X), B))
    for j in range(k):
        np.add.at(Wp, (n, i4p[nn[:, j]].astype(np.int64)), wv[:, j:j + 1] * w4p[nn[:, j]])
    Bg = max(B, int(i4g.max()) + 1)  # the grey cat may have more bones than Pikachu (home2_stand.py adds one)
    Wg = np.zeros((len(X), Bg))
    np.add.at(Wg, (n, i4g.astype(np.int64)), w4g)
    share = Wg.sum(0)
    M = Wg.T @ Wp
    C = (Wg.T @ Qa) / np.maximum(share, 1e-9)[:, None]
    mix = []
    for i in range(Bg):
        if share[i] < 1e-6:
            mix.append([[0, 1.0]])
            continue
        o = np.argsort(-M[i])[:top]
        w = M[i, o] / M[i, o].sum()
        mix.append([[int(a), float(b)] for a, b in zip(o, w) if b > 1e-4])
    return mix, C, share / len(X)


def region_retarget(T, mix, C):
    """Pikachu's bone motion T (F,B,3,4) -> the grey cat's: each grey bone moves rigidly with the blend of Pikachu's
    bones in its mix (rotation: blended quaternions; translation: where the blend carries the bone's centre)"""
    F = T.shape[0]
    q = quat_from_mat(T[..., :3])
    G = np.zeros((F, len(mix), 3, 4))
    for i, mx in enumerate(mix):
        ks = np.array([a for a, _ in mx])
        ws = np.array([b for _, b in mx])
        qk = q[:, ks]
        sg = np.sign((qk * qk[:, :1]).sum(-1, keepdims=True))
        sg[sg == 0] = 1
        qb = (ws[None, :, None] * sg * qk).sum(1)
        R = mat_from_quat(qb / np.linalg.norm(qb, axis=-1, keepdims=True))
        Tk = T[:, ks]
        moved = (ws[None, :, None] * (Tk[..., :3] @ C[i] + Tk[..., 3])).sum(1)
        G[:, i, :, :3] = R
        G[:, i, :, 3] = moved - R @ C[i]
    return G


def chain_len(j, idx):
    return float(sum(np.linalg.norm(j[b] - j[a]) for a, b in zip(idx[:-1], idx[1:])))


def main():
    page_dir, uv_npz, skel_npz, paths_dir, out = sys.argv[1:6]
    os.makedirs(out, exist_ok=True)
    am = json.load(open(os.path.join(page_dir, "a2s_cat.json")))
    A2 = read_blobs(os.path.join(page_dir, "a2s_cat.bin"), am)
    P, B = A2["pos"].astype(np.float64), am["bones"]
    rec = A2["bones"].astype(np.float64).reshape(-1, B, 3, 4)
    wm = json.load(open(os.path.join(page_dir, "a2s_walks.json")))
    walks = read_blobs(os.path.join(page_dir, "a2s_walks.bin"), wm)["bones"].astype(np.float64).reshape(-1, B, 3, 4)
    pm = json.load(open(os.path.join(paths_dir, "paths.json")))
    heading = read_blobs(os.path.join(paths_dir, "paths.bin"), pm)["heading"].astype(np.float64).reshape(-1, 3, 3)
    z = np.load(uv_npz)
    Q, faces2, uv2 = z["pos"].astype(np.float64), z["faces"].astype(np.int64), z["uv"].astype(np.float32)
    sk = np.load(skel_npz)
    parents = sk["parents"].astype(np.int64)
    assert len(parents) == B and len(sk["skin_idx"]) == len(Q), (len(parents), B, len(sk["skin_idx"]), len(Q))
    assert all(parents[i] < i for i in range(B))
    print("Pikachu: %d vertices, %d recorded + %d walk frames; grey cat: %d vertices; %d bones" % (
        len(P), len(rec), len(walks), len(Q), B), flush=True)

    # ---- Pikachu's rest joints, from its bone motion
    pj, pres = joints_from_bones(rec[::4], parents)
    print("Pikachu's joints from its bone motion: residual median %.2f mm, worst %.2f mm (%s)" % (
        1000 * np.median(pres), 1000 * pres.max(), NAMES[int(pres.argmax())]), flush=True)

    # ---- REBASE=auto: the grey cat's canonical rest pose is bent (body curved sideways, paws tucked together), so use
    # instead the frame of its own video whose pose is closest to Pikachu's rest (best rigid fit, on every 6th vertex)
    tree_p = cKDTree(P)
    rebase, Qc = None, Q  # Qc: the canonical rest mesh, for its own video's stretch
    if os.environ.get("REBASE", "auto") in ("auto", "level"):
        own = sk["own"].astype(np.float64)
        i4g, w4g = sk["skin_idx"].astype(np.int64), sk["skin_w"].astype(np.float64)
        Psub = P[::3]
        tree_s = cKDTree(Psub)
        scores = []
        for f in range(len(own)):
            X = lbs(Q, i4g, w4g, own[f])[::6]
            mp_, mq_ = Psub.mean(0), X.mean(0)
            ep_ = np.linalg.eigh(np.cov((Psub - mp_).T))[1][:, ::-1]
            eq_ = np.linalg.eigh(np.cov((X - mq_).T))[1][:, ::-1]
            s0_ = np.sqrt(((Psub - mp_) ** 2).sum(1).mean() / ((X - mq_) ** 2).sum(1).mean())
            best = 1e9
            for sg in ([1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]):
                R0 = ep_ @ np.diag(sg) @ eq_.T
                if np.linalg.det(R0) < 0:
                    R0 = ep_ @ np.diag(np.array(sg) * [1, 1, -1]) @ eq_.T
                best = min(best, icp(X, Psub, s0_, R0, mp_ - s0_ * R0 @ mq_, tree_s)[3])
            scores.append(best)
        # only frames where the fitted pose clearly stands on four separate paws (front pair ahead of the hind pair);
        # REBASE=level (feet and pose check): also square, as Pikachu's rest is: the four paws level (within 2.5 cm),
        # each pair side by side (at least 4 cm apart sideways, at most 5 cm apart front to back); all frames tried
        e_q = edges_of(faces2)
        square = os.environ.get("REBASE", "auto") == "level"
        cand = [f for f in np.argsort(scores)[:len(scores) if square else 40]]
        rebase = None
        for f in cand:  # best fit first; check four paws in Pikachu's frame after that frame's own rigid fit
            X = lbs(Q, i4g, w4g, own[f])
            mp_, mq_ = P.mean(0), X.mean(0)
            ep_ = np.linalg.eigh(np.cov((P - mp_).T))[1][:, ::-1]
            eq_ = np.linalg.eigh(np.cov((X - mq_).T))[1][:, ::-1]
            s0_ = np.sqrt(((P - mp_) ** 2).sum(1).mean() / ((X - mq_) ** 2).sum(1).mean())
            fits_ = []
            for sg in ([1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]):
                R0 = ep_ @ np.diag(sg) @ eq_.T
                if np.linalg.det(R0) < 0:
                    R0 = ep_ @ np.diag(np.array(sg) * [1, 1, -1]) @ eq_.T
                fits_.append(icp(X[::4], P[::3], s0_, R0, mp_ - s0_ * R0 @ mq_, tree_s))
            s_, R_, t_ = min(fits_, key=lambda z_: z_[3])[:3]
            tips = paw_tips(s_ * X @ R_.T + t_, e_q)
            sq = ""
            if tips is not None and square:
                t4 = np.array(tips)  # sorted front to back: front pair, then hind pair
                spread = t4[:, 1].max() - t4[:, 1].min()
                apart = min(abs(t4[0, 0] - t4[1, 0]), abs(t4[2, 0] - t4[3, 0]))
                stagger = max(abs(t4[0, 2] - t4[1, 2]), abs(t4[2, 2] - t4[3, 2]))
                sq = ", level within %.1f cm, pairs %.1f cm apart sideways, %.1f cm front to back" % (
                    100 * spread, 100 * apart, 100 * stagger)
                if not (spread <= 0.025 and apart >= 0.04 and stagger <= 0.05):
                    tips = None
            print("  own frame %d: fit %.1f mm, four paws %s%s" % (f, 1000 * scores[f], "yes" if tips else "no", sq), flush=True)
            if tips is not None:
                rebase = int(f)
                break
        assert rebase is not None, "no frame with four clear paws"
        print("rebase: own frame %d of %d (raw frame %d), fit %.1f mm (best of all frames %.1f mm, worst %.1f mm, "
              "median %.1f mm)" % (rebase, len(own), int(sk["own_frames"][rebase]), 1000 * scores[rebase], 1000 * min(scores),
                                                  1000 * max(scores), 1000 * np.median(scores)), flush=True)
        Q = lbs(Q, i4g, w4g, own[rebase])

    # ---- line the grey cat's rest mesh up with Pikachu's (one scale, rotation, shift), as before
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
    s, R, t, fd = min(fits, key=lambda f: f[3])[:4]
    scale_mode = os.environ.get("SCALE", "fit")
    qj = s * sk["joints"].astype(np.float64) @ R.T + t
    leg_p = np.mean([chain_len(pj, v) for v in LEGS.values()])
    leg_q = np.mean([chain_len(qj, v) for v in LEGS.values()])
    if scale_mode == "legs":  # grow or shrink about the hips' midpoint so the legs are as long as Pikachu's
        c, k = qj[[17, 21]].mean(0), leg_p / leg_q
        s, t = s * k, c + k * (t - c)
        qj = s * sk["joints"].astype(np.float64) @ R.T + t
    Qa = s * Q @ R.T + t
    print("fit: scale %.4f m per model unit (%s), mesh distance %.1f mm" % (s, scale_mode, 1000 * fd), flush=True)
    jd = np.linalg.norm(qj - pj, axis=1)
    print("rest joints, grey vs Pikachu after the fit (mm): " + ", ".join(
        "%s %.0f" % (n, 1000 * d) for n, d in zip(NAMES, jd)), flush=True)
    sw = np.array([0, 1, 2, 3, 8, 9, 10, 11, 4, 5, 6, 7, 12, 13, 14, 15, 16, 21, 22, 23, 24, 17, 18, 19, 20])
    print("left and right swapped would be %.0f mm on average (as is %.0f mm)" % (
        1000 * np.linalg.norm(qj[sw] - pj, axis=1).mean(), 1000 * jd.mean()), flush=True)

    # ---- the retarget constants, and bone lengths for the record. MODE region (used): each grey bone follows the
    # flesh of Pikachu its own vertices correspond to; dir / none (tried first): bone i follows Pikachu's bone i,
    # turned onto Pikachu's bone direction (dir) or by the same turn from rest (none)
    align_mode = os.environ.get("MODE", "region")
    A = rest_alignment(pj, qj, parents) if align_mode == "dir" else np.broadcast_to(np.eye(3), (B, 3, 3)).copy()
    print("mode: %s" % align_mode, flush=True)
    i4, w4 = sk["skin_idx"].astype(np.int64), sk["skin_w"].astype(np.float64)
    if align_mode == "region":
        lams = [float(x) for x in os.environ.get("NR_LAMS", "30 10 3").split()]
        X = nonrigid(Qa, faces2, P, A2["faces"].astype(np.int64), tree_p, lams) if lams else Qa
        mix, C, share = region_mix(Qa, X, i4, w4, P, A2["skin_idx"], A2["skin_w"], B)
        for i in range(B):
            if share[i] > 0.005:
                print("  grey %-10s (%4.1f%% of vertices) <- %s" % (NAMES[i], 100 * share[i], ", ".join(
                    "%s %.0f%%" % (NAMES[a], 100 * b) for a, b in mix[i][:3])), flush=True)
        do_rt = lambda T: region_retarget(T, mix, C)  # noqa: E731
    else:
        do_rt = lambda T: retarget(T, A, qj, parents)  # noqa: E731
    ang = np.degrees(np.arccos(np.clip((np.trace(A, axis1=1, axis2=2) - 1) / 2, -1, 1)))
    print("rest-pose correction per bone (degrees): " + ", ".join("%s %.0f" % (n, a) for n, a in zip(NAMES, ang)),
          flush=True)
    legs = {k: [chain_len(pj, v), chain_len(qj, v)] for k, v in LEGS.items()}
    spine = [chain_len(pj, [17, 0, 1, 2, 3]), chain_len(qj, [17, 0, 1, 2, 3])]
    tail = [chain_len(pj, [12, 13, 14, 15, 16]), chain_len(qj, [12, 13, 14, 15, 16])]
    print("lengths (m), Pikachu vs grey: legs %s; hip to head %.3f vs %.3f; tail %.3f vs %.3f" % (
        {k: [round(a, 3), round(b, 3)] for k, (a, b) in legs.items()}, *spine, *tail), flush=True)

    # ---- checks: stretch against the grey cat's own motion, and where its lowest point sits against Pikachu's
    e1, e2 = edges_of(A2["faces"].astype(np.int64)), edges_of(faces2)
    fr = np.linspace(0, len(rec) - 1, CHECK_FRAMES).round().astype(int)
    fw = np.linspace(0, len(walks) - 1, CHECK_FRAMES).round().astype(int)
    G_rec, G_walk = do_rt(rec[fr]), do_rt(walks[fw])
    st_p, st_g, st_own, low_gap = [], [], [], []
    for k, f in enumerate(fr):
        p1 = lbs(P, A2["skin_idx"], A2["skin_w"], rec[f])
        p2 = lbs(Qa, i4, w4, G_rec[k])
        st_p.append(stretch(P, p1, e1))
        st_g.append(stretch(Qa, p2, e2))
        down = heading[f][1]
        low_gap.append((p2 @ down).max() - (p1 @ down).max())
    for k in range(len(fw)):
        st_g.append(stretch(Qa, lbs(Qa, i4, w4, G_walk[k]), e2))
    own = sk["own"].astype(np.float64)
    for f in np.linspace(0, len(own) - 1, min(CHECK_FRAMES, len(own))).round().astype(int):
        st_own.append(stretch(Qc, lbs(Qc, i4, w4, own[f]), e2))
    st_p, st_g, st_own = np.concatenate(st_p), np.concatenate(st_g), np.concatenate(st_own)
    q_own, q_g = float(np.percentile(st_own, 99)), float(np.percentile(st_g, 99))
    ratio = q_g / max(q_own, 1e-9)
    check = {
        "scale_mode": scale_mode, "align_mode": align_mode, "rebase_own_frame": rebase, "fit_scale_m_per_unit": float(s), "fit_mean_distance_mm": 1000 * fd,
        "pikachu_joint_residual_mm_median": 1000 * float(np.median(pres)),
        "rest_joint_distance_mm": dict(zip(NAMES, (1000 * jd).round(1).tolist())),
        "rest_correction_deg": dict(zip(NAMES, ang.round(1).tolist())),
        "legs_m_pikachu_grey": legs, "hip_to_head_m_pikachu_grey": spine, "tail_m_pikachu_grey": tail,
        "skin_top4_cover": float(sk["skin_cover"]),
        "stretch_p99_grey_own_video": q_own, "stretch_p99_grey_retargeted": q_g, "stretch_p99_pikachu": float(np.percentile(st_p, 99)),
        "ratio_retargeted_to_own": ratio,
        "readout": "moves cleanly" if ratio <= 1.5 else ("some stretching" if ratio <= 3 else "tears"),
        "share_edges_over_30pct_own": float((st_own > np.log(1.3)).mean()),
        "share_edges_over_30pct_retargeted": float((st_g > np.log(1.3)).mean()),
        "lowest_point_vs_pikachu_mm_median": 1000 * float(np.median(low_gap)),
        "lowest_point_vs_pikachu_mm_p10_p90": [1000 * float(np.percentile(low_gap, 10)), 1000 * float(np.percentile(low_gap, 90))],
        "grey_size_m": float(np.linalg.norm(Qa.max(0) - Qa.min(0))), "pikachu_size_m": float(np.linalg.norm(P.max(0) - P.min(0))),
    }
    json.dump(check, open(os.path.join(out, "home2_retarget.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in check.items() if not isinstance(v, dict)}, indent=1), flush=True)

    name, tex = os.environ.get("NAME", "a2s_cat2"), os.environ.get("TEX", "cat2_tex_photo.jpg")  # the whole Body tab cat: a2s_cat3
    meta, nbytes = pack(os.path.join(out, name + ".bin"), [
        ("faces", faces2.astype(np.uint32)), ("pos", Qa.astype(np.float32)), ("uv", uv2),
        ("skin_idx", i4.astype(np.uint8)), ("skin_w", w4.astype(np.float32))])
    json.dump({"blobs": meta, "bytes": nbytes, "vertices": len(Qa), "bones": B, "frames": 0,
               "texture": tex, "units": "metres, Pikachu's (Agent-to-Sim's) own frame",
               **({"mix": mix, "centre": np.round(C, 6).tolist(),
                   "from": "the grey and white cat's own rest mesh and skinning, lined up with Pikachu's; each of its "
                           "bones moves rigidly with the blend of Pikachu's bones that move the flesh its vertices "
                           "correspond to (rotation: blended quaternions; translation: where the blend carries the "
                           "bone's centre)"} if align_mode == "region" else
                  {"parents": parents.tolist(), "joints": np.round(qj, 6).tolist(),
                   "align": np.round(A.reshape(B, 9), 6).tolist(),
                   "from": "bone i follows Pikachu's bone i (rotation x align), joints by forward kinematics"})},
              open(os.path.join(out, name + ".json"), "w"))
    print("%s: %d bytes" % (name, nbytes), flush=True)

    # ---- picture: rest and 4 recorded + 2 walk frames, side and top; Pikachu orange, grey cat blue, skeletons drawn
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    v18 = sum(v["frames"] for v in pm["videos"][:17]) + 900  # the ship rule's moments (walk 3 = walk run 2 from 1360)
    v5 = sum(v["frames"] for v in pm["videos"][:4]) + 200
    sel = [("video 18, 90 s", rec[v18]), ("video 5, 20 s", rec[v5]), ("walk 3, 15 s", walks[1510]),
           ("walk 3, 5 s", walks[1410]), ("recorded %d" % fr[16], rec[fr[16]]), ("walk frame %d" % fw[30], walks[fw[30]])]
    cols = [("rest", None, None)] + [(n, T_, do_rt(T_[None])[0]) for n, T_ in sel]
    fig, axes = plt.subplots(2, len(cols), figsize=(4.2 * len(cols), 8))
    for c, (title, Tp, Tg) in enumerate(cols):
        if Tp is None:
            p1, p2, j1 = P, Qa, pj
        else:
            p1, p2 = lbs(P, A2["skin_idx"], A2["skin_w"], Tp), lbs(Qa, i4, w4, Tg)
            j1 = np.einsum("bij,bj->bi", Tp[:, :, :3], pj) + Tp[:, :, 3]
        for r, (a, b) in enumerate(((2, 1), (2, 0))):
            ax = axes[r, c]
            ax.scatter(p1[::4, a], p1[::4, b], s=0.3, c="tab:orange", alpha=0.3)
            ax.scatter(p2[::4, a], p2[::4, b], s=0.3, c="tab:blue", alpha=0.3)
            for i, p in enumerate(parents):
                if p >= 0:
                    ax.plot(j1[[p, i], a], j1[[p, i], b], c="darkred", lw=1)
            ax.set_aspect("equal")
            if r == 0:
                ax.invert_yaxis()
                ax.set_title(title)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "home2_retarget.png"), dpi=70)


if __name__ == "__main__":
    main()
