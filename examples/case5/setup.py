import numpy as np
import porepy as pp
import porepy_mesh_factory as pmf

def gb(mesh_kwargs):
    network = pmf.main.generate("flow_benchmark_3d_case_2", only_network=True)
    c_min, c_max = -0.1, 1.1
    network.domain = {"xmin": c_min, "xmax": c_max,
                      "ymin": c_min, "ymax": c_max,
                      "zmin": c_min, "zmax": c_max,}

    # create the grid bucket
    return network.mesh(mesh_kwargs)

def data(spe10):
    tol = 1e-8
    for g, d in spe10.gb:
        # Set up parameters with a fake permeability
        perm = pp.SecondOrderTensor(np.ones(g.num_cells))

        b_faces = g.tags["domain_boundary_faces"].nonzero()[0]

        # define the labels and values for the boundary faces
        labels = np.array(["dir"] * b_faces.size)
        bc_val = g.face_centers[1]/spe10.full_physdims[1] * 1e7

        bc = pp.BoundaryCondition(g, b_faces, labels)

        f = 0*g.cell_volumes

        parameters = {
            "second_order_tensor": perm,
            "bc": bc,
            "bc_values": bc_val,
            "source": f,
        }
        pp.initialize_default_data(g, d, "flow", parameters)
