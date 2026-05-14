import os, sys
import subprocess, random
import networkx as nx
from parsers import parse_dmx
import numpy as np
from pathlib import Path


def build_graph(n, arcs):
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    for (u, v, _, _, _) in arcs:
        G.add_edge(u, v)
    return G


def is_feasible(n, arcs, b):
    # 1) balance
    if not np.isclose(b.sum(), 0):
        return False

    G = build_graph(n, arcs)

    supply_nodes = np.where(b > 0)[0]
    demand_nodes = set(np.where(b < 0)[0])

    # 2) directed reachability
    for s in supply_nodes:
        reachable = nx.descendants(G, s)
        if not (reachable & demand_nodes):
            return False

    return True

def generate_instance(pargen_path, netgen_path, qfcgen_path,
                      m, k, out_dir):

    rho = 3  # high density
    cf = random.choice(["a", "b"])
    cq = random.choice(["a", "b"])
    scale = random.choice(["s", "ns"])

    base = f"netgen-{m}-{rho}-{k}-{cf}-{cq}-{scale}"
    par_file = os.path.join(out_dir, base + ".par")
    dmx_file = os.path.join(out_dir, base + ".dmx")
    qfc_file = os.path.join(out_dir, base + ".qfc")

    # Step 1: pargen
    cp = subprocess.run(
        [pargen_path, str(m), str(rho), str(k), cf, cq, scale],
        cwd=out_dir,
        check=True
    )
    print(cp)
    # Step 2: netgen
    with open(par_file, "r") as fin, open(dmx_file, "w") as fout:
        subprocess.run([netgen_path], stdin=fin, stdout=fout, check=True)

    # Step 3: qfcgen
    subprocess.run([qfcgen_path, dmx_file], check=True)

    return dmx_file, qfc_file


def generate_feasible_instances(
    pargen_path,
    netgen_path,
    qfcgen_path,
    out_dir,
    m=1000,
    num_instances=5,
    max_attempts=50
):
    os.makedirs(out_dir, exist_ok=True)

    feasible_files = []
    k = 1
    attempts = 0

    while len(feasible_files) < num_instances and attempts < max_attempts:
        attempts += 1

        try:
            dmx_file, qfc_file = generate_instance(
                pargen_path, netgen_path, qfcgen_path,
                m, k, out_dir
            )

            _,E,b,_,_, arcs = parse_dmx(dmx_file, return_arcs=True)
            n = E.shape[0]

            if is_feasible(n, arcs, b):
                print(f"Feasible instance found: {dmx_file}")
                feasible_files.append((dmx_file, qfc_file))
            else:
                print(f"Infeasible instance discarded: {dmx_file}")
                os.remove(dmx_file)
                if os.path.exists(qfc_file):
                    os.remove(qfc_file)

        except Exception as e:
            print(f"Error during generation: {e}")

        k += 1

    return feasible_files



if __name__ == '__main__':

    if len(sys.argv) != 4:
        print(f"Usage: python3 -m instances.instance_generator [pargen_path] [qfcgen_path] [netgen_path]")
        exit(1)

    pargen_path = Path(sys.argv[1]).absolute()
    qfcgen_path = Path(sys.argv[2]).absolute()
    netgen_path = Path(sys.argv[3]).absolute()

    out_dir = "./instances/myinstances"

    files = generate_feasible_instances(
        pargen_path,
        netgen_path,
        qfcgen_path,
        out_dir,
        m=35,
        num_instances=3
    )

    print("Generated feasible instances:")
    for f in files:
        print(f)

