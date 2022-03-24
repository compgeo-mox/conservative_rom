import numpy as np
import scipy.sparse as sps

import porepy as pp
import pygeon as pg

class HodgeSolver():

    def __init__(self, gb, discr, data=None, data_key="flow", if_check=True):
        self.gb = gb
        self.data = data
        self.data_key = data_key

        self.grad = pg.grad(gb)
        self.curl = pg.curl(gb)
        self.div  = pg.div(gb)

        # Testing
        if if_check:
            assert (self.curl * self.grad).nnz == 0
            assert (self.div * self.curl).nnz == 0

        self.mass = pg.hdiv_mass(gb, discr, data)

        #BBt = self.div*self.h_scaling*self.div.T
        self.BBt = sps.linalg.splu((self.div*self.div.T).tocsc())

        #gb: g2 g1 mg
        #extract the mass matrices and put them on the (block) diagonal

        #h_scaling = np.mean(g.cell_diameters())**(g.dim - 2)

    def solve(self, linalg_solve = sps.linalg.spsolve):
        q_f = self.step1()
        sigma = self.step2(q_f, linalg_solve)
        return self.step3(q_f, sigma)

    def step1(self):
        f = self.assemble_source()

        p_f = self.BBt.solve(f)
        #q_f = h_scaling*self.div.T*p_f
        q_f = self.div.T*p_f
        return q_f

    def step2(self, q_f, linalg_solve = sps.linalg.spsolve):
        A = self.curl.T*self.mass*self.curl
        A += self.grad*self.grad.T
        b = - self.curl.T*self.mass*q_f

        if np.allclose(A * np.ones(A.shape[1]), 0):
            R = sps.eye(A.shape[1] - 1, A.shape[1])        
        else:
            #Create restriction that removes tip dofs
            R = pg.numerics.differentials.zero_tip_dofs(self.gb, 2).tocsr()
            R = R[R.indices, :]

        sol = linalg_solve(R*A*R.T, R*b)

        return R.T * sol

    def step3(self, q_f, sigma):
        q = q_f + self.curl*sigma

        #p = sps.linalg.spsolve(BBt, h_scaling*div*M*q)
        p = self.BBt.solve(self.div*self.mass*q)
        return q, p

    def assemble_source(self):
        if isinstance(self.gb, pp.Grid):
            return self.data[pp.PARAMETERS][self.data_key]["source"]

        else: # gb is a GridBucket
            f = []
            for _, d in self.gb:
                f.append(d[pp.PARAMETERS]["flow"]["source"])

            return np.concatenate(f)
