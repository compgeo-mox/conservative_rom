import numpy as np
import scipy.sparse as sps
import porepy as pp
import pygeon as pg
import scipy.stats.qmc as qmc
from hodge_solver import HodgeSolver


class Hodge_offline:
    def __init__(self, hs: HodgeSolver, n_snaps=10):
        self.hs = hs

        samples = qmc.LatinHypercube(2).random(n_snaps)
        l_bounds = np.array([-2, -10])
        u_bounds = np.array([2, 10])

        self.mu_params = qmc.scale(samples, l_bounds, u_bounds)
        self.mu_params[:, 0] = 10.0 ** self.mu_params[:, 0]

        self.S = self.generate_snapshots()
        self.U, self.Sigma, _ = np.linalg.svd(self.S, full_matrices=False)

        self.U = self.truncate_U()

    def generate_snapshots(self):
        snapshots = [self.solve_one_instance(mu) for mu in self.mu_params]
        return np.column_stack(snapshots)

    def solve_one_instance(self, mu):
        hs_temp = self.scaled_copy(mu)

        q_f = hs_temp.step1()
        sigma = hs_temp.step2(q_f)

        return sigma

    def scaled_copy(self, mu):
        hs_temp = self.hs.copy()

        self.adjust_data(hs_temp, mu)

        return hs_temp

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

    def truncate_U(self, threshold=1e-6):
        I = np.cumsum(self.Sigma**2) / np.sum(self.Sigma**2)
        N = np.argmax(I >= 1 - threshold**2)

        if N + 1 == I.size:
            raise Warning("You should have more snapshots")

        return self.U[:, : N + 1]


class Hodge_online:
    def __init__(self, h_off: Hodge_offline):
        self.h_off = h_off

    def solve(self, mu, linalg_solve=...):
        hs_temp = self.h_off.scaled_copy(mu)

        hs_temp.create_restriction = self.create_restriction

        q, p = hs_temp.solve()

        return q, p

    def create_restriction(self):
        return sps.csr_matrix(self.h_off.U.T)
