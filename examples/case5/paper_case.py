import numpy as np

import porepy as pp
import pygeon as pg

import sys

sys.path.insert(0, "../../src/")
sys.path.insert(0, "src/")
from hodge_solver import HodgeSolver
from hodge_rom import *
import reference

import setup

"""
    Case 3 is a fixed-dimensional case in 3D using MFEM
"""

def main():
    gb = setup.gb()
    import pdb; pdb.set_trace()
    pg.compute_geometry(gb)

    import pdb; pdb.set_trace()
    setup.data(gb)

    discr = pp.RT0("flow")

    hs = HodgeSolver(gb, discr)

    h_off = Hodge_offline_case3(hs)
    h_off.save("./results/")
    h_on = Hodge_online(h_off)

    n_modes = h_off.U.shape[1]
    print("n_modes =", n_modes)
    h_off.plot_singular_values()

    # Comparison to a known solution
    mu = spe10.read_perm(31) #########################
    hs_full = h_off.scaled_copy(mu)

    q_ref, p_ref = reference.full_saddlepoint_system(hs_full)
    q, p = h_on.solve(mu)

    reference.dim_check(q, p, q_ref, p_ref, hs_full)

class Hodge_offline_case3(Hodge_offline):
    def generate_samples(self):

        n_snaps = 3
        l_bounds = np.array([-5])
        u_bounds = np.array([ 5])
        samples = qmc.LatinHypercube(l_bounds.size).random(n_snaps)

        mu_params = qmc.scale(samples, l_bounds, u_bounds)
        mu_params[0] = 10.0 ** mu_params[0]

        return mu_params

    def adjust_data(self, hs, mu):
        perm = pp.SecondOrderTensor(mu[0])

        for g, d in hs.gb:
            if g.dim == 2 or g.dim == 1:
                d["parameters"]["flow"]["second_order_tensor"] = perm

        hs.mass = hs.compute_mass_matrix()
        hs.f = hs.assemble_source()
        hs.g = hs.assemble_rhs()

    def truncate_U(self, threshold=1e-7):
        N = np.argmax(self.Sigma <= threshold)

        return self.U[:, :N]

if __name__ == "__main__":
    main()
