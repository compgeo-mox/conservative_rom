import numpy as np

import porepy as pp
import pygeon as pg

import sys
sys.path.insert(0, "../../src/")
sys.path.insert(0, "src/")
from hodge_solver import HodgeSolver
import reference

import setup

"""
    Case 2 is a mixed-dimensional case in 2D.
"""

def main():
    gb = setup.gb()
    pg.compute_geometry(gb)

    setup.data(gb)

    discr = pp.RT0("flow") # MVEM

    hs = HodgeSolver(gb, discr)
    q, p = hs.solve()

    # q_ref, p_ref = reference.mixed_dim(data_key, gb, discr)
    q_ref, p_ref = reference.full_saddlepoint_system(hs)
    reference.dim_check(q, p, q_ref, p_ref, hs)

if __name__ == "__main__":
    np.set_printoptions(linewidth=9999)
    main()