import numpy as np
import scipy.sparse as sps

import porepy as pp
import pygeon as pg


class HodgeSolver():

    def __init__(self, gb, discr, data=None, if_check=True):
        self.gb = gb
        self.data = data

        self.grad = pg.grad(gb)
        self.curl = pg.curl(gb)
        self.div = pg.div(gb)

        # Testing
        if if_check:
            assert (self.curl * self.grad).nnz == 0
            assert (self.div * self.curl).nnz == 0

        self.mass = pg.hdiv_mass(gb, discr, data)

        #BBt = self.div*self.h_scaling*self.div.T
        BBt = self.div*self.div.T
        self.BBt = sps.linalg.splu(BBt.tocsc())

        #h_scaling = np.mean(g.cell_diameters())**(g.dim - 2)

    def solve(self, linalg_solve=sps.linalg.spsolve):
        q_f = self.step1()
        sigma = self.step2(q_f, linalg_solve)
        return self.step3(q_f, sigma)

    def step1(self):
        f = self.assemble_source()

        p_f = self.BBt.solve(f)
        #q_f = h_scaling*self.div.T*p_f
        q_f = self.div.T*p_f
        return q_f

    def step2(self, q_f, linalg_solve=sps.linalg.spsolve):
        A = self.curl.T*self.mass*self.curl
        A += self.grad*self.grad.T
        b = self.curl.T*(self.assemble_rhs() - self.mass*q_f)

        # Check if we're in the Dirichlet case with n = 2
        if np.allclose(A * np.ones(A.shape[1]), 0):
            # Create restriction that removes last dof
            R = sps.eye(A.shape[1] - 1, A.shape[1])
        else:  # All other cases
            # Create restriction that removes tip dofs
            R = pg.remove_tip_dofs(self.gb, 2)

        sol = linalg_solve(R*A*R.T, R*b)

        return R.T * sol

    def step3(self, q_f, sigma):
        q = q_f + self.curl*sigma

        #p = sps.linalg.spsolve(BBt, h_scaling*div*M*q)
        p = self.BBt.solve(self.div*(self.mass*q - self.assemble_rhs()))
        return q, p

    def assemble_source(self):
        if isinstance(self.gb, pp.Grid):
            return self.data[pp.PARAMETERS]["flow"]["source"]

        else:  # gb is a GridBucket
            f = []
            for _, d in self.gb:
                f.append(d[pp.PARAMETERS]["flow"]["source"])

            return np.concatenate(f)

    def assemble_rhs(self):
        if isinstance(self.gb, pp.Grid):
            raise NotImplementedError

        else:  # gb is a GridBucket
            rhs = []
            for g, d in self.gb:
                bc_values = d[pp.PARAMETERS]["flow"]["bc_values"].copy()
                b_faces = np.where(g.tags["domain_boundary_faces"])[0]
                signs = [g.cell_faces.tocsr()[face, :].data[0]
                         for face in b_faces]

                bc_values[b_faces] *= -np.array(signs)

                rhs.append(bc_values)

            return np.concatenate(rhs)
