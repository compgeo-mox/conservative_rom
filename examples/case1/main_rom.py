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

    h_off = Hodge_offline_case1(hs)
    h_on = Hodge_online(h_off)

    n_modes = h_off.U.shape[1]
    print("n_modes =", n_modes)

    mu = np.array([1e-1, 1.0])
    hs_full = h_off.scaled_copy(mu)

    q_ref, p_ref = reference.full_saddlepoint_system(hs_full)
    q, p = h_on.solve(mu)

    # print("Singular values")
    # print(np.reshape(h_off.Sigma, (-1,)))

    reference.dim_check(q, p, q_ref, p_ref, hs_full)


class Hodge_offline_case1(Hodge_offline):
    def generate_samples(self):

        n_snaps = 10
        samples = qmc.LatinHypercube(2).random(n_snaps)
        l_bounds = np.array([-2, -10])
        u_bounds = np.array([2, 10])

        mu_params = qmc.scale(samples, l_bounds, u_bounds)
        mu_params[:, 0] = 10.0 ** mu_params[:, 0]

        return mu_params

    def adjust_data(self, hs, mu):

        n = hs.gb.dim_max()

        def perm_field(g):
            K = np.ones(g.cell_centers.shape[1])
            K[g.cell_centers[n - 1, :] > 0.5] = mu[0]
            return pp.SecondOrderTensor(K)

        def source(g):
            return g.cell_volumes * mu[1]

        for g, d in hs.gb:
            d["parameters"]["flow"]["second_order_tensor"] = perm_field(g)
            d["parameters"]["flow"]["source"] = source(g)

        hs.mass = hs.compute_mass_matrix()
        hs.f = hs.assemble_source()
        # hs.g = hs.assemble_rhs()


if __name__ == "__main__":
    np.set_printoptions(linewidth=9999)
    [main(N) for N in [10]]
