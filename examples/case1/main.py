import numpy as np
import scipy.sparse as sps
import time
import pyamg

import porepy as pp
import pygeon as pg

import sys
sys.path.insert(0, "src/")
from hodge_solver import HodgeSolver

"""
    Case 1 is a fixed-dimensional case, designed for testing
    MFEM and MVEM in 2D or 3D.
"""


def reference_solution(data_key, g, data, discr):
    A, b_flow = discr.assemble_matrix_rhs(g, data)

    rhs_discr = pp.DualScalarSource(data_key)

    rhs_discr.discretize(g, data)
    _, b_rhs = rhs_discr.assemble_matrix_rhs(g, data)

    qp = sps.linalg.spsolve(A, b_flow+b_rhs)

    # Extract the flux and pressure from the solution
    q = discr.extract_flux(g, qp, data)
    p = discr.extract_pressure(g, qp, data)

    return q, p

def main(N=2):
    data_key = 'flow'
    # 2D
    g = pp.StructuredTriangleGrid([N]*2, [1]*2)
    # g = pp.CartGrid([N]*2, [1]*2)

    # 3D
    # g = pp.StructuredTetrahedralGrid([N]*3, [1]*3)
    # g = pp.CartGrid([N]*3, [1]*3)

    # Set up grid bucket consisting of one grid
    g.compute_geometry()
    gb = pp.meshing.grid_list_to_grid_bucket([[g]])
    setup_data(gb)

    pg.compute_edges(gb)

    discr = pp.RT0(data_key)
    # discr = pp.MVEM(data_key)

    hs = HodgeSolver(gb, discr)

    # def linalg_solve(A, b):
    #     amg = pyamg.ruge_stuben_solver(A.tocsr())
    #     return amg.solve(b, tol=1e-14)

    # q, p = hs.solve(linalg_solve)
    q, p = hs.solve()

    # verification
    data = gb.node_props(g)
    q_ref, p_ref = reference_solution(data_key, g, data, discr)

    f = data[pp.PARAMETERS][data_key]['source']

    print("Pressure error: {:.2E}".format(np.linalg.norm(p-p_ref)))
    print("Flux error:     {:.2E}".format(np.linalg.norm(q-q_ref)))
    print("Mass loss:      {:.2E}".format(np.linalg.norm(hs.div*q - f)))


def setup_data(gb):
    for g, d in gb:
        # Set up parameters
        perm = pp.SecondOrderTensor(
            kxx=4*np.ones(g.num_cells), kyy=np.ones(g.num_cells), kxy=np.ones(g.num_cells))
        b_faces = g.tags["domain_boundary_faces"].nonzero()[0]
        bc = pp.BoundaryCondition(g, b_faces, ["dir"]*b_faces.size)
        bc_val = np.zeros(g.num_faces)
        f = g.cell_volumes

        parameters = {"second_order_tensor": perm,
                    "bc": bc, "bc_values": bc_val, "source": f}
        data_key = "flow"
        pp.initialize_default_data(g, d, data_key, parameters)


if __name__ == "__main__":
    np.set_printoptions(linewidth=9999)
    [main(N) for N in [6]]
