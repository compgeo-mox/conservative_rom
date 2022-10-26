import numpy as np

import porepy as pp
import pygeon as pg

import sys

sys.path.insert(0, "../../src/")
sys.path.insert(0, "src/")
from hodge_solver import HodgeSolver
from hodge_rom import *
import scipy.stats.qmc as qmc
import reference

import setup

"""
    Case 2 is a mixed-dimensional case in 2D using MVEM
"""

random_seed = 0


def main():
    # create the grid bucket
    gb = setup.gb(0.05)
    pg.compute_geometry(gb)

    setup.data(gb)

    discr = pp.MVEM("flow")

    hs = HodgeSolver(gb, discr)

    h_off = Hodge_offline_case2(hs, random_seed)
    h_off.save("./results/")
    h_on = Hodge_online(h_off)

    n_modes = h_off.U.shape[1]
    print("n_modes =", n_modes)
    h_off.plot_singular_values(1e-7)

    dofs = np.zeros(4, dtype=int)
    dofs[0] = gb.num_cells() + gb.num_faces()
    dofs[1] = gb.num_cells()
    dofs[2] = hs.curl.shape[1]
    dofs[3] = n_modes

    print(dofs)

    # Comparison to a known solution
    mu = [0, 1, 1e-4, 1e4]
    hs_full = h_off.scaled_copy(mu)

    q_ref, p_ref = reference.full_saddlepoint_system(hs_full)
    q, p = h_on.solve(mu)

    reference.check(q, p, q_ref, p_ref, hs_full)


class Hodge_offline_case2(Hodge_offline):
    def generate_samples(self, random_seed=None):
        n_snaps = 110
        l_bounds = np.array([0, 0, -5, 3])
        u_bounds = np.array([1, 1, -3, 5])

        samples = qmc.LatinHypercube(l_bounds.size, seed=random_seed).random(n_snaps)
        mu_params = qmc.scale(samples, l_bounds, u_bounds)
        mu_params[:, -2:] = 10.0 ** mu_params[:, -2:]

        return mu_params

    def adjust_data(self, hs, mu):
        alpha_0 = np.zeros(3)
        alpha_0[:2] = mu[:2]
        k_low = mu[2]
        k_high = mu[3]

        setup.set_perm(hs.gb, k_low, k_high)

        def bc_values(g):
            b_faces = g.tags["domain_boundary_faces"]
            f_centers = g.face_centers[:, b_faces]

            values = np.zeros(g.num_faces)
            values[b_faces] = np.dot(alpha_0, f_centers)
            return values

        for g, d in hs.gb:
            d["parameters"]["flow"]["bc_values"] = bc_values(g)

        hs.mass = hs.compute_mass_matrix()
        # hs.f = hs.assemble_source()
        hs.g = hs.assemble_rhs()

    def truncate_U(self, threshold=1e-7):
        N = np.argmax(self.Sigma <= threshold)

        return self.U[:, :N]


if __name__ == "__main__":
    main()
