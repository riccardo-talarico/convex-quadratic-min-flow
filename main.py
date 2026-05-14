from parsers import parse_dmx, parse_qfc, transform_bounds
from solvers import generic_lp_solver, successive_shortest_paths
from frank_wolfe import *
import os
import networkx as nx

import numpy as np

#TODO: forse sta roba si può togliere
def repair_instance(E, b, lb, ub, c, penalty_cost=1e6):
    n, m = E.shape

    # Build graph
    G = nx.DiGraph()
    G.add_nodes_from(range(n))

    for j in range(m):
        u = np.where(E[:, j] == -1)[0][0]
        v = np.where(E[:, j] == 1)[0][0]
        G.add_edge(u, v)

    supply_nodes = np.where(b > 0)[0]
    demand_nodes = np.where(b < 0)[0]

    new_cols = []
    new_lb = []
    new_ub = []
    new_c = []

    for s in supply_nodes:
        reachable = nx.descendants(G, s)

        if not any(d in reachable for d in demand_nodes):
            # pick any demand node
            t = demand_nodes[0]

            # create new column for E
            col = np.zeros(n)
            col[s] = -1
            col[t] = 1

            new_cols.append(col)

            # bounds: allow enough flow
            new_lb.append(0.0)
            new_ub.append(abs(b[s]))  # enough to route supply

            # high penalty cost so solver avoids it if possible
            new_c.append(penalty_cost)

            print(f"Added repair arc: {s} -> {t}")

    # If no fixes needed
    if not new_cols:
        return E, b, lb, ub, c

    # Stack new columns
    E_new = np.hstack([E, np.column_stack(new_cols)])
    lb_new = np.concatenate([lb, np.array(new_lb)])
    ub_new = np.concatenate([ub, np.array(new_ub)])
    c_new = np.concatenate([c, np.array(new_c)])

    return E_new, b, lb_new, ub_new, c_new



if __name__ == '__main__':
    filenames = os.listdir("instances/myinstances")
    for i in range(0, len(filenames),3):
        print(f"instances/{filenames[i]}")
        q, E, b, lb, ub = parse_dmx(f"instances/myinstances/{filenames[i]}")
        fixed_cost, quadratic = parse_qfc(f"instances/myinstances/{filenames[i+2]}")

        
        q,E,b,ub = transform_bounds(q,E,b,lb,ub)

        cmf = ConvexMinFlow(quadratic, q, E, b, ub)
        assert cmf.check_init()
        
        y_ssp = successive_shortest_paths(q,E,b,lb,ub)
        print(y_ssp)
        y_lp = generic_lp_solver(q, E, b, lb, ub)
        print(y_lp)
        
    