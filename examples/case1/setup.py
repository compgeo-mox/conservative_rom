import numpy as np
import porepy as pp
import pygeon as pg

def mdg(N):
    # 3D
    sd = pp.StructuredTetrahedralGrid([N] * 3, [1] * 3)
    mdg = pp.meshing.subdomains_to_mdg([[sd]])
    pg.convert_from_pp(mdg)
    mdg.compute_geometry()

    return mdg

def data(mdg):
    for sd, data in mdg.subdomains(return_data=True):
        # Set up parameters
        perm = pp.SecondOrderTensor(np.ones(sd.num_cells))
        b_faces = sd.tags["domain_boundary_faces"].nonzero()[0]
        bc = pp.BoundaryCondition(sd, b_faces, ["dir"] * b_faces.size)
        bc_val = np.zeros(sd.num_faces)
        # bc_val[b_faces] = np.sin(2 * np.pi * sd.face_centers[1, b_faces])

        f = np.ones(sd.num_cells)

        parameters = {
            "second_order_tensor": perm,
            "bc": bc,
            "bc_values": bc_val,
            "source": f,
        }
        pp.initialize_default_data(sd, data, "flow", parameters)
