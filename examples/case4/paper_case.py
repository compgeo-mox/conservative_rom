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
from spe10 import Spe10

"""
    Case 2 is a fixed-dimensional case in 3D using MFEM
"""

def main():
    # create the grid bucket from a specific layer
    spe10 = Spe10()

    spe10.create_gb()
    pg.compute_geometry(spe10.gb)

    setup.data(spe10)

    discr = pp.MVEM("flow")

    hs = HodgeSolver(spe10.gb, discr)

    h_off = Hodge_offline_case2(hs, spe10)
    h_off.save("./results/")
    h_on = Hodge_online(h_off)

    n_modes = h_off.U.shape[1]
    print("n_modes =", n_modes)
    h_off.plot_singular_values()

    # Comparison to a known solution
    mu = spe10.read_perm(31)
    hs_full = h_off.scaled_copy(mu)

    q_ref, p_ref = reference.full_saddlepoint_system(hs_full)
    q, p = h_on.solve(mu)

    reference.dim_check(q, p, q_ref, p_ref, hs_full)

class Hodge_offline_case2(Hodge_offline):
    def __init__(self, hs, spe10):
        self.spe10 = spe10
        super().__init__(hs)

    def generate_samples(self):
        snaps_layer = np.arange(30)
        return [self.spe10.read_perm(s) for s in snaps_layer]

    def adjust_data(self, hs, mu):
        perm = pp.SecondOrderTensor(kxx=mu[:, 0], kyy=mu[:, 1], kzz=mu[:, 2])

        for g, d in hs.gb:
            d["parameters"]["flow"]["second_order_tensor"] = perm

        hs.mass = hs.compute_mass_matrix()
        hs.f = hs.assemble_source()
        hs.g = hs.assemble_rhs()

    def truncate_U(self, threshold=1e-7):
        N = np.argmax(self.Sigma <= threshold)

        return self.U[:, :N]


if __name__ == "__main__":
    main()
