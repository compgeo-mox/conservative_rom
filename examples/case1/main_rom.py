import numpy as np

import porepy as pp
import pygeon as pg

import sys

sys.path.insert(0, "../../src/")
from hodge_solver import HodgeSolver
from hodge_rom import *
import reference

import setup

"""
    Case 1 is a fixed-dimensional case, designed for testing
    MFEM and MVEM in 2D or 3D.
    Here, the second step is replaced by a RBM
"""


def main(N=2):
    gb = setup.gb(N)
    pg.compute_geometry(gb)

    setup.data(gb)

    discr = pp.RT0("flow")  # MVEM

    hs = HodgeSolver(gb, discr)

    h_off = Hodge_offline(hs, n_snaps=10)
    h_on = Hodge_online(h_off)

    n_modes = h_off.U.shape[1]
    print([n_modes, h_off.Sigma[n_modes - 1]])

    mu = np.array([1e-1, 1.0])
    hs_full = h_off.scaled_copy(mu)

    q_ref, p_ref = reference.full_saddlepoint_system(hs_full)
    q, p = h_on.solve(mu)

    print(np.linalg.norm(p - p_ref))

    print("Singular values")
    print(np.reshape(h_off.Sigma, (-1,)))

    reference.dim_check(q, p, q_ref, p_ref, hs_full)


if __name__ == "__main__":
    np.set_printoptions(linewidth=9999)
    [main(N) for N in [10]]
