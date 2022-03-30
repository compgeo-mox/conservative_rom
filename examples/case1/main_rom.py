import numpy as np

import porepy as pp
import pygeon as pg

import sys
sys.path.insert(0, "../../src/")
sys.path.insert(0, "src/")
from hodge_solver import HodgeSolver
import reference

import setup

"""
    Case 1 is a fixed-dimensional case, designed for testing
    MFEM and MVEM in 2D or 3D.
"""

def main(N=2):
    gb = setup.gb(N)
    pg.compute_geometry(gb)

    setup.data(gb)

    ################################################################################
    # pod benchmark:
    ################################################################################

    n_snap_to_generate = 50
    n_snap_to_use = 6
    #n_modes_to_use = {'all': 2}
    do_monolithic = True

    offline = pg.rom.offline_test.OfflineTest()
    offline.remove_old_data()
    offline.generate_snapshots(n_snap_to_generate) # from analytical solution

    offline.load_snapshots(n_snap_to_use, shuffle=False)
    offline.compute_svd(do_monolithic, save=False)
    offline.plot_singular_values() # not necessary
    offline.truncate_U(save_all_svd_matrices=True)

    Phi = offline.assemble_phi() # and save phi

    online = OnlineTest() 
    A, b = online.assemble_full_order_A_rhs()

    # not necessary:
    sol_analytical, sol_fom = online.compute_fom_solution(A, b) # analytical solution + A, b assembling for random mu_params

    sol_reduced = online.compute_reduced_solution(A, b, Phi)
    sol_reconstructed = Phi@sol_reduced

    mse_err = np.sum( (sol_reconstructed - sol_fom)**2 )/sol_fom.size 
    myprint('mse_err')
    myprint('sol_analytical')
    myprint('sol_fom')
    myprint('sol_reconstructed')

    plt.show()

    discr = pp.RT0("flow") # MVEM

    hs = HodgeSolver(gb, discr)

    # import pyamg
    # def linalg_solve(A, b):
    #     amg = pyamg.ruge_stuben_solver(A.tocsr())
    #     return amg.solve(b, tol=1e-14)

    # q, p = hs.solve(linalg_solve)
    q, p = hs.solve()

    # verification
    # data = gb.node_props(g)
    # q_ref, p_ref = reference.equi_dim(data_key, g, data, discr)
    q_ref, p_ref = reference.full_saddlepoint_system(hs)

    reference.dim_check(q, p, q_ref, p_ref, hs)

if __name__ == "__main__":
    np.set_printoptions(linewidth=9999)
    [main(N) for N in [6]]