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
    N *= 1#4
    mdg = setup.mdg(N)

    setup.data(mdg)

    discr = pg.RT0("flow")

    hs = HodgeSolver(mdg, discr)

    dofs = np.zeros(3, dtype=int)
    dofs[0] = np.sum(hs.div.shape)
    dofs[1] = hs.div.shape[0]
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

    import pdb; pdb.set_trace()
    #reference.check(q, p, q_ref, p_ref, hs_full)


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

        def perm_field(sd):
            omega_0 = np.abs(sd.cell_centers[2, :] - 0.375) <= 0.125
            omega_0 += np.abs(sd.cell_centers[2, :] - 0.875) <= 0.125

            K = np.ones(sd.num_cells)
            K[omega_0] = K_0
            return pp.SecondOrderTensor(K)

        def source(sd):
            return sd.cell_volumes * f_0

        def bc_values(sd):
            b_faces = sd.tags["domain_boundary_faces"]
            f_centers = sd.face_centers[:, b_faces]

            values = np.zeros(sd.num_faces)
            values[b_faces] = np.dot(alpha_0, f_centers)
            return values

        for sd, data in hs.mdg.subdomains(return_data=True):
            data["parameters"]["flow"]["second_order_tensor"] = perm_field(sd)
            data["parameters"]["flow"]["source"] = source(sd)
            data["parameters"]["flow"]["bc_values"] = bc_values(sd)

        hs.mass = hs.compute_mass_matrix()
        hs.f = hs.assemble_source()
        hs.g = hs.assemble_rhs()

    def truncate_U(self, threshold=1e-7):
        N = np.argmax(self.Sigma <= threshold)

        return self.U[:, :N]


if __name__ == "__main__":
    main()
