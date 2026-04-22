import numpy as np
import heapq
from pulp import *
from typing import Any

# min{cx : Ex=b, lb <= x <= ub}
def successive_shortest_paths(
        c: np.ndarray, 
        E: np.ndarray, 
        b: np.ndarray,
        lb:np.ndarray, 
        ub:np.ndarray
        ):
    
    n, m = E.shape

    arcs = build_arcs_list(E,m)

    # Initialize flow with lower bounds
    x = lb.copy()

    # Adjust supplies due to lower bounds
    b_res = b.copy().astype(np.float64)
    for k, (i, j) in enumerate(arcs):
        b_res[i] -= lb[k]
        b_res[j] += lb[k]

    # Residual capacities
    cap_fwd = ub - lb
    cap_bwd = lb.copy()  # initially zero backward capacity

    # Node potentials (for reduced costs)
    pi = np.zeros(n)

    # Main loop
    supply_nodes = set(np.where(b_res > 1e-12)[0])
    demand_nodes = set(np.where(b_res < -1e-12)[0])

    while supply_nodes:

        s = next(iter(supply_nodes))

        # shortest path P = parent
        dist, parent = shortest_path(s)
        t = find_reachable_demand_node(dist,demand_nodes)

        # Update potentials
        for i in range(n):
            if dist[i] < np.inf:
                pi[i] += dist[i]

        # Determine augmenting flow: 
        # delta = min{ e(s), -e(t), min{u_ij: ij in P}}
        delta = min(b_res[s], -b_res[t])
        
        v = t
        while v != s:
            u, k, direction = parent[v]
            if direction == +1:
                delta = min(delta, cap_fwd[k])
            else:
                delta = min(delta, cap_bwd[k])
            v = u

        # Augment flow
        v = t
        while v != s:
            u, k, direction = parent[v]
            if direction == +1:
                x[k] += delta
                cap_fwd[k] -= delta
                cap_bwd[k] += delta
            else:
                x[k] -= delta
                cap_fwd[k] += delta
                cap_bwd[k] -= delta
            v = u

        # Update supplies
        b_res[s] -= delta
        b_res[t] += delta

        if abs(b_res[s]) < 1e-12:
            supply_nodes.remove(s)
        if abs(b_res[t]) < 1e-12:
            demand_nodes.remove(t)

    return x


def shortest_path(s, n, arcs, cap_fwd, c, pi, cap_bwd):
    
    dist = np.full(n, np.inf)
    parent = [None] * n
    dist[s] = 0
    pq = [(0, s)]

    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue

        for k, (i, j) in enumerate(arcs):
            # Forward arc
            if i == u and cap_fwd[k] > 0:
                rc = c[k] + pi[i] - pi[j]
                if dist[j] > dist[i] + rc:
                    dist[j] = dist[i] + rc
                    parent[j] = (u, k, +1)
                    heapq.heappush(pq, (dist[j], j))

            # Backward arc
            if j == u and cap_bwd[k] > 0:
                rc = -c[k] + pi[j] - pi[i]
                if dist[i] > dist[j] + rc:
                    dist[i] = dist[j] + rc
                    parent[i] = (u, k, -1)
                    heapq.heappush(pq, (dist[i], i))
    return dist, parent

def build_arcs_list(E:np.ndarray, m:int):
    """
    Builds an arc list from the arc-node incidence matrix.

    Args:
        E (np.ndarray): Arc-node incidence matrix of shape (n,m).
        m (int): number of edges of the graph.
    
    Returns:
        List[Tuple[int,int]]: list of edges, saved as tuples [from,to]
    """
    arcs = []
    for k in range(m):
        tail = np.where(E[:, k] == -1)[0][0]
        head = np.where(E[:, k] == 1)[0][0]
        arcs.append((tail, head))

def find_reachable_demand_node(dist,demand_nodes):
    t = None
    for j in demand_nodes:
        if dist[j] < np.inf:
            t = j
            break

    if t is None:
        raise ValueError("Problem is infeasible")
    return t

def generic_lp_solver(
        linear_constraints,
        incidence_matrix,
        node_flow_constraints,
        lower_bound,
        upper_bound,
        solver: Any | None = None
        ):
    n, m = incidence_matrix.shape

    # Define problem
    prob = LpProblem("min_flow", LpMinimize)

    # Decision variables x_j for each arc j
    x = [
        LpVariable(f"x_{j}", lowBound=lower_bound[j], upBound=upper_bound[j])
        for j in range(m)
    ]

    # Objective: c^T x
    prob += lpSum(linear_constraints[j] * x[j] for j in range(m))

    # Constraints: Ex = b (row by row)
    for i in range(n):
        prob += lpSum(incidence_matrix[i, j] * x[j] for j in range(m)) == node_flow_constraints[i], f"node_{i}"
    if not solver:
        prob.solve()
    else:
        prob.solve(solver)

    return np.array([value(x[j]) for j in range(m)])

