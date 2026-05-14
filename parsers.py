import numpy as np

def parse_dmx(filename, return_arcs:bool = False):
    with open(filename, "r") as f:
        lines = f.readlines()

    n = m = None
    arcs = []
    b_dict = {}

    for line in lines:
        line = line.strip()
        if not line or line.startswith("c"):
            continue

        parts = line.split()

        if parts[0] == "p":
            # p min n m
            n = int(parts[2])
            m = int(parts[3])

        elif parts[0] == "n":
            # n i b_i
            i = int(parts[1]) - 1  # convert to 0-based
            b_dict[i] = float(parts[2])

        elif parts[0] == "a":
            # a u v lb ub cost
            u = int(parts[1]) - 1
            v = int(parts[2]) - 1
            lb = float(parts[3])
            ub = float(parts[4])
            cost = float(parts[5])

            arcs.append((u, v, lb, ub, cost))

    # Default b = 0 if not specified
    b = np.zeros(n)
    for i, val in b_dict.items():
        b[i] = -val

    # Build arrays
    m = len(arcs)
    q = np.zeros(m)
    lb = np.zeros(m)
    ub = np.zeros(m)
    E = np.zeros((n, m))

    for j, (u, v, l, ucap, cost) in enumerate(arcs):
        E[u, j] = -1
        E[v, j] = 1
        q[j] = cost
        lb[j] = l
        ub[j] = ucap
        
    if return_arcs:
        return q,E,b,lb,ub,arcs
    else:
        return q, E, b, lb, ub


def parse_qfc(filename):
    with open(filename, "r") as f:
        data = f.read().split()

    m = int(data[0])
    fixed = np.array(list(map(float, data[1:1 + m])))
    quad = np.array(list(map(float, data[1 + m:1 + 2*m])))

    return fixed, quad


def transform_bounds(q, E, b, lb, ub):
    # New capacities
    ub_new = ub - lb

    # Adjust supply
    b_new = b - E @ lb

    # Cost unchanged (constant term ignored)
    return q, E, b_new, ub_new


if __name__=='__main__':
    from utils.convex_min_flow import ConvexMinFlow
    import os

    filenames = os.listdir("instances/1000")
    for i in range(0, len(filenames), 2) :

    
        q, E, b, lb, ub = parse_dmx(f"instances/1000/{filenames[i]}")
        fixed_cost, quadratic = parse_qfc(f"instances/1000/{filenames[i+1]}")

        q,E,b,ub = transform_bounds(q,E,b,lb,ub)

        
        cmf = ConvexMinFlow(quadratic, q, E, b, ub)

        assert cmf.check_init()
    print("ok")