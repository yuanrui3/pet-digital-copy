"""Side check for the feet and pose check (PROTOCOL_CAT.md): which of Pikachu's paws does each of the grey cat's paws
move with, and on which side of the body does each paw's leg join? The grey cat's rest pose (its own video's frame 50)
has its hind paws almost one behind the other, so naming them by x (left or right) may cross them.
usage: python feet_sides.py PAGE_DIR
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feet_check import LMS, PAWS, Page  # noqa: E402
from home2_transfer import edges_of  # noqa: E402


def geodesic_from(X, faces, src):
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import dijkstra
    e = edges_of(faces)
    L = np.linalg.norm(X[e[:, 0]] - X[e[:, 1]], axis=1)
    G = coo_matrix((np.r_[L, L], (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])), shape=(len(X), len(X))).tocsr()
    return dijkstra(G, indices=src, min_only=True)


def main():
    page = Page(sys.argv[1])
    B = page.B
    # each vertex's motion as a mix over Pikachu's bones: Pikachu's own skinning; the grey cat's skinning times its mix
    Bg = len(page.mix)  # the grey cat's bones (may be more than Pikachu's)

    def wmat(idx, w, nb):
        W = np.zeros((len(idx), nb))
        np.add.at(W, (np.arange(len(idx))[:, None], idx), w)
        return W
    Wp = wmat(page.ip, page.wp, B)
    Mg = np.zeros((Bg, B))
    for i, mx in enumerate(page.mix):
        for a, b in mx:
            Mg[i, a] = b
    Wg = wmat(page.ig, page.wg, Bg) @ Mg
    print("which Pikachu paw each grey paw moves with (cosine of their bone mixes; leg = within 8 cm of the paw tip):")
    print("%-14s" % "grey \\ Pikachu" + "".join("%14s" % p for p in PAWS))
    legs = {}
    for nm, X, f, sets in (("p", page.P, page.fp, page.lp), ("g", page.Q, page.fg, page.lg)):
        legs[nm] = [geodesic_from(X, f, s) for s in sets[:4]]
    vp = [Wp[legs["p"][j] < 0.08].mean(0) for j in range(4)]
    for i in range(4):
        vg = Wg[legs["g"][i] < 0.08].mean(0)
        print("%-14s" % PAWS[i] + "".join("%14.2f" % (vg @ v / np.linalg.norm(vg) / np.linalg.norm(v)) for v in vp))
    print("where each leg joins the body (vertices 10 to 14 cm up the leg from the paw tip), x relative to the body's middle:")
    for nm, X in (("Pikachu", page.P), ("grey", page.Q)):
        mid = np.median(X[:, 0])
        d = legs["p" if nm == "Pikachu" else "g"]
        print("  %-8s " % nm + ", ".join("%s %+.3f" % (PAWS[j], X[(d[j] > 0.10) & (d[j] < 0.14), 0].mean() - mid) for j in range(4)))
    print("rest landmarks (y down, z to the tail):")
    for nm, X, sets in (("Pikachu", page.P, page.lp), ("grey", page.Q, page.lg)):
        print("  %-8s " % nm + ", ".join("%s %s" % (k, np.round(X[s].mean(0), 3)) for k, s in zip(LMS, sets)))


if __name__ == "__main__":
    main()
