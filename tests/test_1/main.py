import numpy as np
import pyamg

import porepy as pp
import pygeon as pg

import sys

sys.path.insert(0, "../../src/")
sys.path.insert(0, "src/")
from hodge_solver import HodgeSolver
import reference

import setup

"""
    Case 1 is a fixed-dimensional case, designed for testing
    MFEM and MVEM in 2D or 3D.
"""


def main(N=2):
    gb = setup.gb(N)
    pg.compute_geometry(gb)

    setup.data(gb)

    discr = pp.RT0("flow")  # MVEM

    hs = HodgeSolver(gb, discr)

    # def linalg_solve(A, b):
    #     amg = pyamg.ruge_stuben_solver(A.tocsr())
    #     return amg.solve(b, tol=1e-14)

    # q, p = hs.solve(linalg_solve)
    q, p = hs.solve()

    # verification
    # data = gb.node_props(g)
    # q_ref, p_ref = reference.equi_dim(data_key, g, data, discr)
    q_ref, p_ref = reference.full_saddlepoint_system(hs)

    reference.dim_check(q, p, q_ref, p_ref, hs)


if __name__ == "__main__":
    np.set_printoptions(linewidth=9999)
    [main(N) for N in [6]]
