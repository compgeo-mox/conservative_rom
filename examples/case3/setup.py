import numpy as np
import porepy as pp
import porepy_mesh_factory as pmf


def gb(mesh_kwargs):
    network = pmf.main.generate("flow_benchmark_3d_case_2", only_network=True)
    c_min, c_max = -0.1, 1.1
    network.domain = {
        "xmin": c_min,
        "xmax": c_max,
        "ymin": c_min,
        "ymax": c_max,
        "zmin": c_min,
        "zmax": c_max,
    }

    # create the grid bucket
    return network.mesh(mesh_kwargs)


def data(mdg, data_key="flow"):

    # Thickness of fracture
    aperture = 1e-4
    fracture_perm = 1

    for g, d in mdg.subdomains(return_data=True):
        # The concept of specific volumes accounts for the thickness
        # of the fracture, which is collapsed in the mixed-dimensional
        # model.
        specific_volumes = np.power(aperture, mdg.dim_max() - g.dim)
        # Permeability
        k = np.ones(g.num_cells) * specific_volumes
        if g.dim < mdg.dim_max():
            k *= fracture_perm
        perm = pp.SecondOrderTensor(k)

        # Zero scalar source already integrated in each cell
        f = 0.0 * g.cell_volumes * specific_volumes

        # Boundary conditions
        b_faces = g.tags["domain_boundary_faces"].nonzero()[0]
        bc = pp.BoundaryCondition(g, b_faces, ["dir"] * b_faces.size)
        bc_val = np.zeros(g.num_faces)
        # bc_val[b_faces] = g.face_centers[1, b_faces]

        parameters = {
            "second_order_tensor": perm,
            "source": f,
            "bc": bc,
            "bc_values": bc_val,
        }
        pp.initialize_data(g, d, data_key, parameters)

    for mg, d in mdg.interfaces(return_data=True):
        # Division through aperture/2 may be thought of as taking the gradient, i.e.
        # dividing by the distance from the matrix to the center of the fracture.
        kn = fracture_perm / (aperture / 2)
        pp.initialize_data(mg, d, data_key, {"normal_diffusivity": kn})


# ------------------------------------------------------------------------------#
