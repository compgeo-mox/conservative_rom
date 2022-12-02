import numpy as np
import scipy.sparse as sps

import porepy as pp
import pygeon as pg


class HodgeSolver:
    def __init__(self, mdg, discr, perform_check=True):
        self.mdg = mdg
        self.discr = discr

        self.grad = pg.grad(mdg)
        self.curl = pg.curl(mdg)
        self.div = pg.div(mdg)

        self.cell_mass = pg.cell_mass(mdg)
        self.face_mass = self.compute_mass_matrix()

        self.cell_div = self.cell_mass * self.div

        self.f = self.assemble_source()
        self.g = self.assemble_rhs()

        # Testing
        if perform_check:
            assert (self.curl * self.grad).nnz == 0
            assert (self.div * self.curl).nnz == 0

        self.Ldiv_inv, self.Lcurl, self.Lgrad_inv = self.compute_lumped_matrices()

        BBt = (self.cell_div) * self.Ldiv_inv * (self.cell_div).T
        self.BBt = sps.linalg.splu(BBt.tocsc())

    def compute_mass_matrix(self):
        return pg.face_mass(self.mdg, self.discr)

    def compute_lumped_matrices(self):
        L = np.empty(3, dtype=object)
        L[0] = pg.numerics.innerproducts.lumped_mass_matrix(self.mdg, 1, self.discr)
        L[1:3] = [
            pg.numerics.innerproducts.lumped_mass_matrix(self.mdg, n_minus_k)
            for n_minus_k in [2, 3]
        ]

        L[0].data = 1.0 / L[0].data  # Hdiv
        # if self.mdg.dim_max() == 3:
        #     L[1] = self.compute_lumped_matrix_ridge()  # Hcurl
        L[2].data = 1.0 / L[2].data  # Hgrad

        return L

    def compute_lumped_matrix_ridge(self):
        blocks = []
        for sd in self.mdg.subdomains():
            tangents = sd.nodes * sd.ridge_peaks
            h = np.linalg.norm(tangents, axis=0)

            cell_ridges = np.abs(sd.face_ridges) * np.abs(sd.cell_faces)
            cell_ridges.data[:] = 1.0

            volumes = cell_ridges * sd.cell_volumes

            blocks.append(sps.diags(volumes / (h * h)))

        return sps.block_diag(blocks)

    def solve(self, linalg_solve=sps.linalg.spsolve):
        q_f = self.step1()
        sigma = self.step2(q_f, linalg_solve)
        return self.step3(q_f, sigma)

    def step1(self):
        p_f = self.BBt.solve(self.f)
        q_f = self.Ldiv_inv * (self.cell_div).T * p_f
        return q_f

    def step2(self, q_f, linalg_solve=sps.linalg.spsolve):
        A = self.curl.T * self.face_mass * self.curl
        A += (self.Lcurl * self.grad) * self.Lgrad_inv * (self.Lcurl * self.grad).T
        b = self.curl.T * (self.g - self.face_mass * q_f)

        R = self.create_restriction()

        sol = linalg_solve(R * A * R.T, R * b)

        return R.T * sol

    def step3(self, q_f, sigma):
        q = q_f + self.curl * sigma

        rhs = self.cell_div * self.Ldiv_inv * (self.face_mass * q - self.g)
        p = self.BBt.solve(rhs)
        return q, p

    def create_restriction(self):
        n = self.curl.shape[1]

        # If the constants are in the kernel, then we are in the 2D Dirichlet case
        if np.allclose(n * np.ones(n), 0):
            # Create restriction that removes last dof
            R = sps.eye(n - 1, n)

        else:  # All other cases
            # Create restriction that removes tip dofs
            R = pg.remove_tip_dofs(self.mdg, 2)

        return R

    def assemble_source(self):
        f = []
        for _, data in self.mdg.subdomains(return_data=True):
            f.append(data[pp.PARAMETERS]["flow"]["source"])

        return np.concatenate(f)

    def assemble_rhs(self):
        rhs = []
        for sd, data in self.mdg.subdomains(return_data=True):
            bc_values = data[pp.PARAMETERS]["flow"]["bc_values"].copy()
            b_faces = np.where(sd.tags["domain_boundary_faces"])[0]
            signs = [sd.cell_faces.tocsr()[face, :].data[0] for face in b_faces]

            bc_values[b_faces] *= -np.array(signs)

            rhs.append(bc_values)

        return np.concatenate(rhs)

    def copy(self):
        copy_self = HodgeSolver.__new__(HodgeSolver)

        for str, attr in self.__dict__.items():
            copy_self.__setattr__(str, attr)

        return copy_self
