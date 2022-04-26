import numpy as np

import porepy as pp
import pygeon as pg

import sys

from sympy import true

sys.path.insert(0, "../../src")
sys.path.insert(0, "src/")
from hodge_solver import HodgeSolver
from hodge_rom import *
import reference

import setup

"""
    Case 2 is a mixed-dimensional case in 2D.
    Here, the second step is replaced by a RBM
"""


def main(N=2):
    h = 1.0 / N
    gb = setup.gb(h)
    pg.compute_geometry(gb)

    setup.data(gb)

    discr = pp.RT0("flow")  # MVEM

    hs = HodgeSolver_rom(gb, discr)

    h_off = Hodge_offline_case2(hs)
    h_on = Hodge_online(h_off)

    n_modes = h_off.U.shape[1]
    print("n_modes =", n_modes)

    mu = np.array([1e-2, 5.0])
    hs_full = h_off.scaled_copy(mu)

    import time

    timer = []
    timer.append(time.time())
    q_ref, p_ref = reference.full_saddlepoint_system(hs_full)
    timer.append(time.time())
    q, p = h_on.solve(mu)
    timer.append(time.time())

    print("Speedup factor: ", (timer[1] - timer[0]) / (timer[2] - timer[1]))

    # print("Singular values")
    # print(np.reshape(h_off.Sigma, (-1,)))

    reference.dim_check(q, p, q_ref, p_ref, hs_full)


class HodgeSolver_rom(HodgeSolver):
    def __init__(self, gb, discr, perform_check=True):
        super().__init__(gb, discr, perform_check)
        self.mass_bmats = self.mass
        self.mass = self.assemble_bmat(*self.mass_bmats)

    def compute_mass_matrix(self):

        return pg.numerics.innerproducts.mass_matrix(
            self.gb, self.discr, 1, return_bmat=True
        )

    def assemble_bmat(self, bmat_g, bmat_mg):
        return sps.bmat(bmat_g, format="csc") + sps.bmat(bmat_mg, format="csc")

    def copy(self):
        copy_self = HodgeSolver_rom.__new__(HodgeSolver_rom)

        for str, attr in self.__dict__.items():
            copy_self.__setattr__(str, attr)

        return copy_self


class Hodge_offline_case2(Hodge_offline):
    def generate_samples(self):

        n_snaps = 20
        samples = qmc.LatinHypercube(2).random(n_snaps)
        l_bounds = np.array([-2, -10])
        u_bounds = np.array([2, 10])

        mu_params = qmc.scale(samples, l_bounds, u_bounds)
        mu_params[:, 0] = 10.0 ** mu_params[:, 0]

        return mu_params

    def adjust_data(self, hs, mu):
        bmat_g, bmat_mg = hs.mass_bmats
        bmat_g = bmat_g.copy()
        bmat_mg = bmat_mg.copy()

        for g, d in hs.gb:
            if g.dim < hs.gb.dim_max():
                nn_g = d["node_number"]
                bmat_g[nn_g, nn_g] = bmat_g[nn_g, nn_g] / mu[0]

        for e, d in hs.gb.edges():
            g_up = hs.gb.nodes_of_edge(e)[1]
            nn_g = hs.gb.node_props(g_up, "node_number")

            bmat_mg[nn_g, nn_g] = bmat_mg[nn_g, nn_g] / mu[0]

        hs.mass = hs.assemble_bmat(bmat_g, bmat_mg)
        hs.f = mu[1] * hs.f
        # hs.g = hs.assemble_rhs()


if __name__ == "__main__":
    np.set_printoptions(linewidth=9999)
    [main(N) for N in [2]]
