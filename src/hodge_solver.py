import numpy as np
import scipy.sparse as sps

import porepy as pp
import pygeon as pg


class HodgeSolver:
    def __init__(self, gb, discr, perform_check=True):
        self.gb = gb
        self.discr = discr

        self.grad = pg.grad(gb)
        self.curl = pg.curl(gb)
        self.div = pg.div(gb)

        # Testing
        if perform_check:
            assert (self.curl * self.grad).nnz == 0
            assert (self.div * self.curl).nnz == 0

        self.mass = self.compute_mass_matrix()
        self.f = self.assemble_source()
        self.g = self.assemble_rhs()

        self.Ldiv_inv, self.Lcurl, self.Lgrad_inv = self.compute_lumped_matrices()

        BBt = self.div * self.Ldiv_inv * self.div.T
        self.BBt = sps.linalg.splu(BBt.tocsc())

    def compute_mass_matrix(self):
        return pg.hdiv_mass(self.gb, self.discr)

    def compute_lumped_matrices(self):
        L = [
            pg.numerics.innerproducts.lumped_mass_matrix(self.gb, self.discr, n_minus_k)
            for n_minus_k in [1, 2, 3]
        ]

        L[0].data = 1.0 / L[0].data
        L[2].data = 1.0 / L[2].data

        return L

    def solve(self, linalg_solve=sps.linalg.spsolve):
        q_f = self.step1()
        sigma = self.step2(q_f, linalg_solve)
        return self.step3(q_f, sigma)

    def step1(self):
        p_f = self.BBt.solve(self.f)
        q_f = self.Ldiv_inv * self.div.T * p_f
        return q_f

    def step2(self, q_f, linalg_solve=sps.linalg.spsolve):
        A = self.curl.T * self.mass * self.curl
        A += (self.Lcurl * self.grad) * self.Lgrad_inv * (self.Lcurl * self.grad).T
        b = self.curl.T * (self.g - self.mass * q_f)

        R = self.create_restriction()

        sol = linalg_solve(R * A * R.T, R * b)

        return R.T * sol

    def step3(self, q_f, sigma):
        q = q_f + self.curl * sigma

        p = self.BBt.solve(self.div * self.Ldiv_inv * (self.mass * q - self.g))
        return q, p

    def create_restriction(self):
        n = self.curl.shape[1]

        # If the constants are in the kernel, then we are in the 2D Dirichlet case
        if np.allclose(n * np.ones(n), 0):
            # Create restriction that removes last dof
            R = sps.eye(n - 1, n)

        else:  # All other cases
            # Create restriction that removes tip dofs
            R = pg.remove_tip_dofs(self.gb, 2)

        return R

    def assemble_source(self):
        f = []
        for _, d in self.gb:
            f.append(d[pp.PARAMETERS]["flow"]["source"])

        return np.concatenate(f)

    def assemble_rhs(self):
        rhs = []
        for g, d in self.gb:
            bc_values = d[pp.PARAMETERS]["flow"]["bc_values"].copy()
            b_faces = np.where(g.tags["domain_boundary_faces"])[0]
            signs = [g.cell_faces.tocsr()[face, :].data[0] for face in b_faces]

            bc_values[b_faces] *= -np.array(signs)

            rhs.append(bc_values)

        return np.concatenate(rhs)

    def copy(self):
        copy_self = HodgeSolver.__new__(HodgeSolver)

        for str, attr in self.__dict__.items():
            copy_self.__setattr__(str, attr)

        return copy_self
