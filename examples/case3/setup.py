import numpy as np
import porepy as pp

def gb():

    # The fractures are specified by their vertices, stored in a numpy array
    f_1 = pp.Fracture(0.9*np.array([[0, 0, 0, 0], [1, 0, -1, 0], [0, 1, 0, -1]]))
    f_2 = pp.Fracture(0.9*np.array([[-1, 0, 1, 0], [0, 0, 0, 0], [0, 1, 0, -1]]))
    f_3 = pp.Fracture(0.9*np.array([[1, 0, -1, 0], [0, 1, 0, -1], [0, 0, 0, 0]]))

    # Also define the domain
    domain = {'xmin': -1, 'xmax': 1, 'ymin': -
              1, 'ymax': 1, 'zmin': -1, 'zmax': 1}

    # Define a 3d FractureNetwork, similar to the 2d one
    network = pp.FractureNetwork3d([f_1], domain=domain)
    mesh_args = {'mesh_size_frac': 0.3, 'mesh_size_min': 0.2}

    return network.mesh(mesh_args)

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
