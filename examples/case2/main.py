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
    mdg = setup.gb(0.01)

    pg.convert_from_pp(mdg)
    mdg.compute_geometry()

    setup.data(mdg)

    discr = pg.MVEM("flow")

    hs = HodgeSolver(mdg, discr)

    h_off = Hodge_offline_case2(hs, random_seed)
    h_off.save("./results/")
    h_on = Hodge_online(h_off)

    n_modes = h_off.U.shape[1]
    print("n_modes =", n_modes)
    h_off.plot_singular_values(1e-7)

    dofs = np.zeros(4, dtype=int)
    dofs[0] = mdg.num_subdomain_cells() + mdg.num_subdomain_faces()
    dofs[1] = mdg.num_subdomain_cells()
    dofs[2] = hs.curl.shape[1]
    dofs[3] = n_modes

    print(dofs)

    # Comparison to a known solution
    num_sim = 20
    for k_p_exp in np.linspace(3, 5, num_sim):
        for k_m_exp in np.linspace(-5, -3, num_sim):
            print("perform simulation for the following parameters", k_p_exp, k_m_exp)

            mu = [0, 1, np.power(10, k_m_exp), np.power(10, k_p_exp)]
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

        setup.set_perm(hs.mdg, k_low, k_high)

        def bc_values(sd):
            b_faces = sd.tags["domain_boundary_faces"]
            f_centers = sd.face_centers[:, b_faces]

            values = np.zeros(sd.num_faces)
            values[b_faces] = np.dot(alpha_0, f_centers)
            return values

        for sd, d in hs.mdg.subdomains(return_data=True):
            d["parameters"]["flow"]["bc_values"] = bc_values(sd)

        hs.face_mass = hs.compute_mass_matrix()
        # hs.f = hs.assemble_source()
        hs.g = hs.assemble_rhs()

    def truncate_U(self, threshold=1e-7):
        N = np.argmax(self.Sigma <= threshold)

        return self.U[:, :N]


if __name__ == "__main__":
    main()
