import numpy as np
import scipy.sparse as sps

import porepy as pp
import pygeon as pg

"""
    Case 3 is a mixed-dimensional case in 3D.
"""

def main():

    # The fractures are specified by their vertices, stored in a numpy array
    f_1 = pp.Fracture(np.array([[0, 0, 0, 0], [1, 0, -1, 0], [0, 1, 0, -1]]))
    f_2 = pp.Fracture(np.array([[-1, 0, 1, 0], [0, 0, 0, 0], [0, 1, 0, -1]]))
    f_3 = pp.Fracture(np.array([[1, 0, -1, 0], [0, 1, 0, -1], [0, 0, 0, 0]]))

    # Also define the domain
    domain = {'xmin': -2, 'xmax': 2, 'ymin': -2, 'ymax': 2, 'zmin': -2, 'zmax': 2}

    # Define a 3d FractureNetwork, similar to the 2d one
    network = pp.FractureNetwork3d([f_1, f_2, f_3], domain=domain)
    mesh_args = {'mesh_size_frac': 0.3, 'mesh_size_min': 0.2}

    # Generate the mixed-dimensional mesh
    gb = network.mesh(mesh_args)
    pg.compute_edges(gb)

    div = pg.div(gb)
    curl = pg.curl(gb)
    grad = pg.grad(gb)

    assert (div * curl).nnz == 0
    assert (curl * grad).nnz == 0


if __name__ == "__main__":
    np.set_printoptions(linewidth=9999)
    main()