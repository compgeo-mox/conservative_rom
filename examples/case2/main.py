import numpy as np
import scipy.sparse as sps
import time

import porepy as pp
import pygeon as pg

def main():

    file_name = "network.csv"
    domain = {"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1}
    network = pp.fracture_importer.network_2d_from_csv(file_name, domain=domain)

    # set the mesh size
    mesh_size = 1e-1
    mesh_kwargs = {"mesh_size_frac": mesh_size, "mesh_size_min": mesh_size / 20}

    # create the grid bucket
    gb = network.mesh(mesh_kwargs)
    pg.compute_edges(gb)

if __name__ == "__main__":
    np.set_printoptions(linewidth=9999)
    main()
