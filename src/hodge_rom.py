import numpy as np
import scipy.sparse as sps
from hodge_solver import HodgeSolver


class Hodge_offline:
    def __init__(self, hs: HodgeSolver, random_seed = None):
        self.hs = hs

        self.mu_params = self.generate_samples(random_seed)

        self.S = self.generate_snapshots()
        self.U, self.Sigma, _ = np.linalg.svd(self.S, full_matrices=False)

        self.U = self.truncate_U()

    def generate_samples(self, random_seed = None):
        pass

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
        pass

    def truncate_U(self, threshold=1e-10):
        I = np.cumsum(self.Sigma**2) / np.sum(self.Sigma**2)
        N = np.argmax(1.0 - I <= threshold**2)

        if N >= I.size - 2:
            Warning("You should probably have more snapshots")

        return self.U[:, : N + 1]

    def save(self, str):
        np.savez(str + "saved", S=self.S, Sigma=self.Sigma, U=self.U)

    def plot_singular_values(self, horizontal_line=None):
        import matplotlib as mpl
        import matplotlib.pyplot as plt

        plt.rc("text", usetex=True)
        plt.rc("font", family="serif")
        plt.rc("font", size=25)

        params = {"text.latex.preamble" : r"\usepackage{bm}\usepackage{amsmath}"}
        plt.rcParams.update(params)
        mpl.rcParams["axes.linewidth"] = 1.5

        plt.plot(np.arange(len(self.Sigma)) + 1.0, self.Sigma, marker="o")
        plt.yscale("log")
        plt.title("Singular values")

        plt.grid(True, linestyle="--", alpha=0.5)
        plt.xlabel("Index $i$")
        plt.ylabel("$\sigma_i$")

        if horizontal_line is not None:
            plt.plot(np.arange(len(self.Sigma)) + 1.0, horizontal_line*np.ones(len(self.Sigma)), linestyle="-.")

        plt.savefig("results/singular_values.pdf", bbox_inches="tight")
        plt.gcf().clear()

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
