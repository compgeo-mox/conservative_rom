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

    h_off = Hodge_offline(hs)
    h_on = Hodge_online(h_off, hs)

    mu = np.array([1.0, -10.0, 1.0])
    q, p = h_on.solve(mu)

    q_ref, p_ref = reference.full_saddlepoint_system(hs)

    reference.dim_check(q, p, q_ref, p_ref, hs)

    print("all done")


if __name__ == "__main__":
    np.set_printoptions(linewidth=9999)
    [main(N) for N in [6]]
