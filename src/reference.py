import numpy as np
import scipy.sparse as sps
import porepy as pp
import pygeon as pg


def equi_dim(data_key, g, data, discr):
    A, b_flow = discr.assemble_matrix_rhs(g, data)

    rhs_discr = pp.DualScalarSource(data_key)

    rhs_discr.discretize(g, data)
    _, b_rhs = rhs_discr.assemble_matrix_rhs(g, data)

    qp = sps.linalg.spsolve(A, b_flow + b_rhs)

    # Extract the flux and pressure from the solution
    q = discr.extract_flux(g, qp, data)
    p = discr.extract_pressure(g, qp, data)

    return q, p


def full_saddlepoint_system(hs):
    n_p, n_q = hs.div.shape

    R = pg.remove_tip_dofs(hs.mdg, 1)
    R = sps.block_diag((R, sps.identity(n_p)))

    A = sps.bmat([[hs.mass, -hs.cell_div.T], [hs.cell_div, None]], format="csr")
    b = np.concatenate((hs.g, hs.f))

    sol = R.T * sps.linalg.spsolve(R * A * R.T, R * b)

    return sol[:n_q], hs.cell_proj * sol[n_q:]


def mixed_dim(data_key, mdg, discr, q_name="flux", p_name="pressure"):

    rhs_discr = pp.DualScalarSource(data_key)
    coupling_discr = pp.RobinCoupling(data_key, discr)

    var_name = "flux_pressure"
    mortar_name = "mortar_flux"
    for _, data in self.mdg.subdomains(return_data=True):
        data[pp.PRIMARY_VARIABLES] = {var_name: {"cells": 1, "faces": 1}}
        data[pp.DISCRETIZATION] = {var_name: {"diffusive": discr, "source": rhs_discr}}

    for e, d in gb.edges():
        g1, g2 = gb.nodes_of_edge(e)
        d[pp.PRIMARY_VARIABLES] = {mortar_name: {"cells": 1}}
        d[pp.COUPLING_DISCRETIZATION] = {
            "lambda": {
                g1: (var_name, "diffusive"),
                g2: (var_name, "diffusive"),
                e: (mortar_name_, coupling_discr),
            }
        }

    assembler = pp.Assembler(gb)
    assembler.discretize()

    A, b = assembler.assemble_matrix_rhs()
    qp = sps.linalg.spsolve(A, b)

    # Extract the flux and pressure from the solution
    assembler.distribute_variable(qp)

    q_ref, p_ref = [], []
    for g, d in gb:
        var = d[pp.STATE][var_name]

        d[pp.STATE][p_name + "_ref"] = discr.extract_pressure(g, var, d)
        d[pp.STATE][q_name + "_ref"] = discr.extract_flux(g, var, d)

        for e, d_e in gb.edges_of_node(g):
            if e[0] == g:
                mg = d_e["mortar_grid"]
                d[pp.STATE][q_name + "_ref"] += (
                    mg.signed_mortar_to_primary * d_e[pp.STATE][mortar_name]
                )

        q_ref.append(d[pp.STATE][q_name + "_ref"])
        p_ref.append(d[pp.STATE][p_name + "_ref"])

    # for _, d in gb.edges():
    #     q_ref.append(d[pp.STATE][mortar_name])

    return np.concatenate(q_ref), np.concatenate(p_ref)


def check(q, p, q_ref, p_ref, hs):
    M = pg.cell_mass(hs.gb)

    e_p = np.sqrt(np.dot(p - p_ref, M * (p - p_ref)))
    e_p /= np.sqrt(np.dot(p_ref, M * p_ref))
    e_q = np.sqrt(np.dot(q - q_ref, hs.mass * (q - q_ref)))
    e_q /= np.sqrt(np.dot(q_ref, hs.mass * q_ref))
    e_f = np.linalg.norm(hs.div * q - hs.f)

    print("Pressure error: {:.2E}".format(e_p))
    print("Flux error:     {:.2E}".format(e_q))
    print("Mass loss:      {:.2E}".format(e_f))

def export(gb, discr, q, p, file_name):

    pressure = "pressure"
    flux = "flux"
    mortar = "mortar"

    permeability = "permeability"
    flux_P0 = "flux_P0"

    shift_q = 0
    shift_p = 0
    for g, d in gb:
        d[pp.STATE] = {flux: q[shift_q:(shift_q+g.num_faces)],
                       pressure: p[shift_p:(shift_p+g.num_cells)],
                       permeability: d[pp.PARAMETERS]["flow"]["second_order_tensor"].values[0, 0, :]}
        shift_q += g.num_faces
        shift_p += g.num_cells

    for e, d in gb.edges():
        d[pp.STATE] = {mortar: np.zeros(d["mortar_grid"].num_cells)}

    # export the P0 flux reconstruction
    pp.project_flux(gb, discr, flux, flux_P0, mortar)

    # export the reference solution
    save = pp.Exporter(gb, file_name, folder_name="solution")
    save.write_vtu([pressure, flux_P0, permeability])


