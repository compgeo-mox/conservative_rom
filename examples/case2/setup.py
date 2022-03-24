import numpy as np
import porepy as pp

def gb():

    p = np.array([[0.0, 1.0, 0.0, 1.0], 
                  [0.5, 0.5, 0.0, 1.0]])
    e = np.array([[0, 2], [1, 3]])

    domain = {"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1}
    network = pp.FractureNetwork2d(p, e, domain)

    # set the mesh size
    mesh_size = 1
    mesh_kwargs = {"mesh_size_frac": mesh_size,
                   "mesh_size_min": mesh_size}

    # create the grid bucket
    return network.mesh(mesh_kwargs)

def data(gb, data_key = "flow"):

    # Thickness of fracture
    aperture = 1
    fracture_perm = 1

    for g, d in gb:
        # The concept of specific volumes accounts for the thickness
        # of the fracture, which is collapsed in the mixed-dimensional
        # model.
        specific_volumes = np.power(aperture, gb.dim_max()-g.dim)
        # Permeability
        k = np.ones(g.num_cells) * specific_volumes
        if g.dim < gb.dim_max():
            k *= fracture_perm
        perm = pp.SecondOrderTensor(k)

        # Unitary scalar source already integrated in each cell
        f = 1 * g.cell_volumes * specific_volumes

        # Boundary conditions
        b_faces = g.tags['domain_boundary_faces'].nonzero()[0]
        bc = pp.BoundaryCondition(g, b_faces, ['dir']*b_faces.size)
        bc_val = np.zeros(g.num_faces)
        # bc_val[b_faces] = g.face_centers[1, b_faces]

        parameters = {"second_order_tensor": perm,
                      "source": f, "bc": bc, "bc_values": bc_val}
        pp.initialize_data(g, d, data_key, parameters)

    for e, d in gb.edges():
        mg = d["mortar_grid"]
        # Division through aperture/2 may be thought of as taking the gradient, i.e.
        # dividing by the distance from the matrix to the center of the fracture.
        kn = fracture_perm / (aperture/2)
        pp.initialize_data(mg, d, data_key, {"normal_diffusivity": kn})

    return data_key
