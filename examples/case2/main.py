import numpy as np
import scipy.sparse as sps

import porepy as pp
import pygeon as pg

def main():

    file_name = "network2.csv"
    domain = {"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1}
    network = pp.fracture_importer.network_2d_from_csv(file_name, domain=domain)

    # set the mesh size
    mesh_size = 1
    mesh_kwargs = {"mesh_size_frac": mesh_size, "mesh_size_min": mesh_size / 20}

    # create the grid bucket
    gb = network.mesh(mesh_kwargs)
    pg.compute_edges(gb)

    div  = pg.div(gb)
    curl = pg.curl(gb)
    grad = pg.grad(gb)

    assert (div * curl).nnz == 0
    assert (curl * grad).nnz == 0

if __name__ == "__main__":
    np.set_printoptions(linewidth=9999)
    main()