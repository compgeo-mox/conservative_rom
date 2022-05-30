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
import time

import setup

"""
    Case 3 corresponds to a regular fracture network in 3D
"""
random_seed = 0
start = time.time()


def main():
    # Generate the Porepy gridbucket
    mesh_size = 2 ** (-4)
    mesh_kwargs = {"mesh_size_frac": mesh_size, "mesh_size_min": mesh_size}
    gb = setup.gb(mesh_kwargs)
    setup.data(gb)

    # Compute the edge connectivity
    pg.compute_geometry(gb)

    discr = pp.RT0("flow")

    # Generate a three-step solver
    hs = HodgeSolver(gb, discr)

    # Print the number of dofs
    dofs = np.zeros(3, dtype=int)
    dofs[0] = gb.num_cells() + gb.num_faces()
    dofs[1] = gb.num_cells()
    dofs[2] = hs.curl.shape[1]
    print(dofs)

    # Offline-online split
    h_off = Hodge_offline_case3(hs)
    h_off.save("./results/")
    h_on = Hodge_online(h_off)

    n_modes = h_off.U.shape[1]
    print("n_modes =", n_modes)

    # Plot singular value decomposition
    h_off.plot_singular_values(1e-7)

    # Comparison to a known solution
    mu = [1, 0, 0, 1, 1e4]
    hs_full = h_off.scaled_copy(mu)

    q_ref, p_ref = reference.full_saddlepoint_system(hs_full)
    q, p = h_on.solve(mu)

    reference.check(q, p, q_ref, p_ref, hs_full)


class Hodge_offline_case3(Hodge_offline):
    def generate_samples(self, random_seed):

        n_snaps = 50
        l_bounds = np.array([0, 0, 0, -1, 3])
        u_bounds = np.array([1, 1, 1, 1, 5])
        samples = qmc.LatinHypercube(l_bounds.size, seed=random_seed).random(n_snaps)

        mu_params = qmc.scale(samples, l_bounds, u_bounds)
        mu_params[:, -1] = 10.0 ** mu_params[:, -1]

        return mu_params

    def adjust_data(self, hs, mu):
        print(time.time() - start)

        aperture = 1e-4
        alpha_0 = mu[:3]
        source = mu[3]
        fracture_perm = mu[4]

        for g, d in hs.gb:
            if g.dim < hs.gb.dim_max():
                specific_volumes = np.power(aperture, hs.gb.dim_max() - g.dim)

                k = fracture_perm * np.ones(g.num_cells) * specific_volumes
                d["parameters"]["flow"]["second_order_tensor"] = pp.SecondOrderTensor(k)

                f = specific_volumes * g.cell_volumes * source
                d["parameters"]["flow"]["source"] = f

            else:
                b_faces = g.tags["domain_boundary_faces"]
                f_centers = g.face_centers[:, b_faces]

                values = np.zeros(g.num_faces)
                values[b_faces] = np.dot(alpha_0, f_centers)

                d["parameters"]["flow"]["bc_values"] = values

        for e, d in hs.gb.edges():
            kn = fracture_perm / (aperture / 2)
            d["parameters"]["flow"]["normal_diffusivity"] = kn

        hs.mass = hs.compute_mass_matrix()
        hs.f = hs.assemble_source()
        hs.g = hs.assemble_rhs()

    def truncate_U(self, threshold=1e-7):
        N = np.argmax(self.Sigma <= threshold)

        return self.U[:, :N]


if __name__ == "__main__":
    main()
