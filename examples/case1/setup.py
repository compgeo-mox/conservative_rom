import numpy as np
import porepy as pp

def gb(N):
    # 2D
    g = pp.StructuredTriangleGrid([N]*2, [1]*2)
    # g = pp.CartGrid([N]*2, [1]*2)

    # 3D
    # g = pp.StructuredTetrahedralGrid([N]*3, [1]*3)
    # g = pp.CartGrid([N]*3, [1]*3)

    # Set up grid bucket consisting of one grid
    g.compute_geometry()
    return pp.meshing.grid_list_to_grid_bucket([[g]]), g

def data(gb, data_key="flow"):
    for g, d in gb:
        # Set up parameters
        perm = pp.SecondOrderTensor(
            kxx=4*np.ones(g.num_cells), kyy=np.ones(g.num_cells), kxy=np.ones(g.num_cells))
        b_faces = g.tags["domain_boundary_faces"].nonzero()[0]
        bc = pp.BoundaryCondition(g, b_faces, ["dir"]*b_faces.size)
        bc_val = np.zeros(g.num_faces)
        f = g.cell_volumes

        parameters = {"second_order_tensor": perm,
                    "bc": bc, "bc_values": bc_val, "source": f}
        pp.initialize_default_data(g, d, data_key, parameters)

    return data_key
