import numpy as np
import scipy.sparse as sps
import porepy as pp

def equi_dim(data_key, g, data, discr):
    A, b_flow = discr.assemble_matrix_rhs(g, data)

    rhs_discr = pp.DualScalarSource(data_key)

    rhs_discr.discretize(g, data)
    _, b_rhs = rhs_discr.assemble_matrix_rhs(g, data)

    qp = sps.linalg.spsolve(A, b_flow+b_rhs)

    # Extract the flux and pressure from the solution
    q = discr.extract_flux(g, qp, data)
    p = discr.extract_pressure(g, qp, data)

    return q, p

def mixed_dim(data_key, gb, discr, q_name="flux", p_name="pressure"):

    rhs_discr = pp.DualScalarSource(data_key)
    coupling_discr = pp.RobinCoupling(data_key, discr)

    var_name = "flux_pressure"
    mortar_name = "mortar_flux"
    for g, d in gb:
        d[pp.PRIMARY_VARIABLES] = {var_name: {"cells": 1, "faces": 1}}
        d[pp.DISCRETIZATION] = {var_name: {"diffusive": discr, "source": rhs_discr}}

    for e, d in gb.edges():
        g1, g2 = gb.nodes_of_edge(e)
        d[pp.PRIMARY_VARIABLES] = {mortar_name: {"cells": 1}}
        d[pp.COUPLING_DISCRETIZATION] = {
            "lambda": {
                g1: (var_name, "diffusive"),
                g2: (var_name, "diffusive"),
                e: (mortar_name, coupling_discr),
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

        q_ref.append(d[pp.STATE][q_name + "_ref"])
        p_ref.append(d[pp.STATE][p_name + "_ref"])

    for _, d in gb.edges():
        q_ref.append(d[pp.STATE][mortar_name])

    return np.concatenate(q_ref), np.concatenate(p_ref)

def dim_check(q, p, q_ref, p_ref, hs):
    f = hs.assemble_source()

    # import pdb; pdb.set_trace()
    print("Pressure error: {:.2E}".format(np.linalg.norm(p-p_ref)))
    print("Flux error:     {:.2E}".format(np.sqrt(np.dot(q-q_ref, hs.mass * (q-q_ref)))))
    print("Mass loss:      {:.2E}".format(np.linalg.norm(hs.div*q - f)))
