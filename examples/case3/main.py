import numpy as np
import scipy.sparse as sps

import porepy as pp
import pygeon as pg

import sys
sys.path.insert(0, "../../src/")
sys.path.insert(0, "src/")
from hodge_solver import HodgeSolver
import reference

import setup

"""
    Case 3 is a mixed-dimensional case in 3D.
"""

def main():
    gb = setup.gb()
    pg.compute_edges(gb)

    data_key = setup.data(gb)

    discr = pp.RT0(data_key)  # MVEM

    hs = HodgeSolver(gb, discr)
    q, p = hs.solve()

    q_ref, p_ref = reference.mixed_dim(data_key, gb, discr)
    reference.dim_check(q, p, q_ref, p_ref, hs)


if __name__ == "__main__":
    np.set_printoptions(linewidth=9999)
    main()