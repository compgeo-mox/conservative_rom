import numpy as np
import scipy.sparse as sps
import porepy as pp
import pygeon as pg
import scipy.stats.qmc as qmc
from hodge_solver import HodgeSolver


class Hodge_offline(pg.OfflineComputations):
    def __init__(self, hs: HodgeSolver):
        self.hs = hs

        samples = qmc.LatinHypercube(3).random(10)
        l_bounds = np.array([1e-3, -100, -1])
        u_bounds = np.array([20, 100, 1])

        self.mu_params = qmc.scale(samples, l_bounds, u_bounds)

        self.S = self.generate_snapshots()
        self.U, self.Sigma, _ = np.linalg.svd(self.S, full_matrices=False)

        self.U = self.truncate_U()

    def generate_snapshots(self):
        snapshots = [self.solve_one_instance(mu) for mu in self.mu_params]
        return np.column_stack(snapshots)

    def solve_one_instance(self, mu):
        scale_matrices(self.hs, mu)

        q_f = self.hs.step1()
        sigma = self.hs.step2(q_f)

        undo_scale_matrices(self.hs, mu)

        return sigma

    def truncate_U(self, threshold=1e-2):
        I = np.cumsum(self.Sigma**2) / np.sum(self.Sigma**2)
        N = np.argmax(I >= 1 - threshold**2)

        if N == I.size:
            raise Warning("You should have more snapshots")

        return self.U[:, : N + 1]


class Hodge_online(HodgeSolver):
    def __init__(self, h_off: Hodge_offline, hs: HodgeSolver):
        self.U = h_off.U

        self.gb = hs.gb
        self.data = hs.data

        self.grad = hs.grad
        self.curl = hs.curl
        self.div = hs.div

        self.mass = hs.mass

        self.f = hs.f
        self.g = hs.g

        self.BBt = hs.BBt

    def solve(self, mu, linalg_solve=...):
        scale_matrices(self, mu)
        q, p = super().solve(linalg_solve)
        undo_scale_matrices(self, mu)

        return q, p

    def step2(self, q_f, linalg_solve=None):
        A = self.curl.T * self.mass * self.curl
        A += self.grad * self.grad.T
        b = self.curl.T * (self.g - self.mass * q_f)

        R = sps.csr_matrix(self.U.T)

        sol = sps.linalg.spsolve(R * A * R.T, R * b)

        return R.T * sol


def scale_matrices(hs, mu):
    hs.mass /= mu[0]
    hs.f *= mu[1]
    hs.g *= mu[2]


def undo_scale_matrices(hs, mu):
    return scale_matrices(hs, 1.0 / mu)
