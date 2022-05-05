import numpy as np
import porepy as pp

def data(spe10):
    tol = 1e-8
    for g, d in spe10.gb:
        # Set up parameters with a fake permeability
        perm = pp.SecondOrderTensor(np.ones(g.num_cells))

        b_faces = g.tags["domain_boundary_faces"].nonzero()[0]
        b_face_centers = g.face_centers[:, b_faces]

        # define outflow type boundary conditions
        out_flow = b_face_centers[1] > spe10.full_physdims[1] - tol

        # define inflow type boundary conditions
        in_flow = b_face_centers[1] < 0 + tol

        # define the labels and values for the boundary faces
        labels = np.array(["neu"] * b_faces.size)
        bc_val = np.zeros(g.num_faces)

        labels[in_flow] = "dir"
        labels[out_flow] = "dir"
        bc_val[b_faces[in_flow]] = 0
        bc_val[b_faces[out_flow]] = 1e7

        bc = pp.BoundaryCondition(g, b_faces, labels)

        f = 0*g.cell_volumes

        parameters = {
            "second_order_tensor": perm,
            "bc": bc,
            "bc_values": bc_val,
            "source": f,
        }
        pp.initialize_default_data(g, d, "flow", parameters)
