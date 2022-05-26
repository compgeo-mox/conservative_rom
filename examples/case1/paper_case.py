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
    Case 1 is a fixed-dimensional case in 3D using MFEM
"""

random_seed = 0


def main(N=4):
    N *= 4
    gb = setup.gb(N)
    pg.compute_geometry(gb)

    setup.data(gb)

    discr = pp.RT0("flow")

    hs = HodgeSolver(gb, discr)

    dofs = np.zeros(3, dtype=int)
    dofs[0] = gb.num_cells() + gb.num_faces()
    dofs[1] = gb.num_cells()
    dofs[2] = hs.curl.shape[1]

    print(dofs)

    h_off = Hodge_offline_case1(hs, random_seed)
    h_off.save("./results/")
    h_on = Hodge_online(h_off)

    n_modes = h_off.U.shape[1]
    print("n_modes =", n_modes)
    h_off.plot_singular_values(1e-7)

    # Comparison to a known solution
    mu = np.array([0, 0, 0, 1, 1e3])
    hs_full = h_off.scaled_copy(mu)

    q_ref, p_ref = reference.full_saddlepoint_system(hs_full)
    q, p = h_on.solve(mu)

    reference.dim_check(q, p, q_ref, p_ref, hs_full)

    reference.export(gb, discr, q_ref, p_ref, "solution")

class Hodge_offline_case1(Hodge_offline):
    def generate_samples(self, random_seed=None):

        n_snaps = 44
        l_bounds = np.array([0, 0, 0, -1, -5])
        u_bounds = np.array([1, 1, 1, 1, 5])
        samples = qmc.LatinHypercube(l_bounds.size, seed=random_seed).random(n_snaps)

        mu_params = qmc.scale(samples, l_bounds, u_bounds)
        mu_params[:, -1] = 10.0 ** mu_params[:, -1]

        return mu_params

    def adjust_data(self, hs, mu):
        alpha_0 = mu[:3]
        f_0 = mu[3]
        K_0 = mu[4]

        def perm_field(g):
            omega_0 = np.abs(g.cell_centers[2, :] - 0.375) <= 0.125
            omega_0 += np.abs(g.cell_centers[2, :] - 0.875) <= 0.125

            K = np.ones(g.num_cells)
            K[omega_0] = K_0
            return pp.SecondOrderTensor(K)

        def source(g):
            return g.cell_volumes * f_0

        def bc_values(g):
            b_faces = g.tags["domain_boundary_faces"]
            f_centers = g.face_centers[:, b_faces]

            values = np.zeros(g.num_faces)
            values[b_faces] = np.dot(alpha_0, f_centers)
            return values

        for g, d in hs.gb:
            d["parameters"]["flow"]["second_order_tensor"] = perm_field(g)
            d["parameters"]["flow"]["source"] = source(g)
            d["parameters"]["flow"]["bc_values"] = bc_values(g)

        hs.mass = hs.compute_mass_matrix()
        hs.f = hs.assemble_source()
        hs.g = hs.assemble_rhs()

    def truncate_U(self, threshold=1e-7):
        N = np.argmax(self.Sigma <= threshold)

        return self.U[:, :N]


if __name__ == "__main__":
    main()
