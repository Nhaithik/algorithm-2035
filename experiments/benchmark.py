"""Experiments: measured operation counts and runtimes vs input characteristics.
Run all (writes CSVs to experiments/results/):  python experiments/benchmark.py
The same functions power the web 'Experiments' page with smaller sizes."""
import csv
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algorithms.graphs import prim_heap, prim_matrix, kruskal_mst, floyd_warshall
from algorithms.knapsack import fractional_knapsack, zero_one_knapsack
from algorithms.search import tsp_backtracking, tsp_branch_and_bound, solve_n_queens
from algorithms.structures import tree_workload, heap_workload
from framework.decision_engine import decide
from framework.metrics import timed
from framework.problem_model import random_graph, random_cost_matrix, random_tour_matrix


def mst_experiment(sizes=(50, 100, 200), densities=(0.02, 0.1, 0.5, 1.0)):
    rows = []
    for n in sizes:
        for dens in densities:
            edges = random_graph(n, dens)
            e = len(edges)
            pm, t_pm = timed(prim_matrix, n, edges)
            ph, t_ph = timed(prim_heap, n, edges)
            kr, t_kr = timed(kruskal_mst, n, edges)
            k_ops = kruskal_mst(n, edges, count_ops=True)["ops"]["total"]
            h_ops = prim_heap(n, edges, count_ops=True)["ops"]["total"]
            ops = {"Prim (matrix)": pm["ops"]["total"], "Prim (heap)": h_ops, "Kruskal": k_ops}
            rule = decide("connect_sites", {"V": n, "E": e, "density": e / (n * (n - 1) / 2)})["chosen"]
            rows.append({"V": n, "E": e, "density": round(e / (n * (n - 1) / 2), 3),
                         "prim_matrix_ops": ops["Prim (matrix)"], "prim_heap_ops": h_ops,
                         "kruskal_ops": k_ops,
                         "prim_matrix_ms": round(t_pm, 3), "prim_heap_ms": round(t_ph, 3), "kruskal_ms": round(t_kr, 3),
                         "rule_choice": rule,
                         "fewest_ops": min(ops, key=ops.get),
                         "fastest_ms": min({"Prim (matrix)": t_pm, "Prim (heap)": t_ph, "Kruskal": t_kr}.items(), key=lambda kv: kv[1])[0],
                         "costs_equal": math.isclose(pm["total_cost"], kr["total_cost"]) and math.isclose(ph["total_cost"], kr["total_cost"])})
    return rows


def floyd_experiment(sizes=(10, 20, 40, 80, 120)):
    rows, prev = [], None
    for n in sizes:
        (_, info), t = timed(floyd_warshall, random_cost_matrix(n, 0.5, 1), repeat=1)
        rows.append({"V": n, "ops": info["ops"], "V_cubed": n ** 3, "ops_over_V3": round(info["ops"] / n ** 3, 3),
                     "ms": round(t, 3), "time_ratio_vs_prev": "" if prev is None else round(t / prev[1], 2),
                     "expected_ratio_(V/prevV)^3": "" if prev is None else round((n / prev[0]) ** 3, 2)})
        prev = (n, t)
    return rows


def tsp_experiment(ns=(5, 6, 7, 8, 9, 10, 11, 12)):
    rows = []
    for n in ns:
        m = random_tour_matrix(n, seed=n)
        bb, t_bb = timed(tsp_branch_and_bound, m, repeat=1)
        row = {"n": n, "tours_(n-1)!": math.factorial(n - 1),
               "bb_states": bb["states_generated"], "bb_pruned": bb["states_pruned"], "bb_ms": round(t_bb, 3),
               "bt_states": "", "bt_ms": "", "costs_equal": ""}
        if n <= 10:
            bt, t_bt = timed(tsp_backtracking, m, repeat=1)
            row.update(bt_states=bt["states_generated"], bt_ms=round(t_bt, 3), costs_equal=bt["cost"] == bb["cost"])
        rows.append(row)
    return rows


def tree_experiment(sizes=(1000, 5000, 20000), t=16):
    rows = []
    for n in sizes:
        for r in tree_workload(n, 5, t):
            rows.append({k: r[k] for k in ("structure", "n", "height", "insert_ms", "search_ms",
                                           "search_node_visits", "search_comparisons", "rebalance_ops", "valid")})
    return rows


def heap_experiment(sizes=(1000, 4000), ratios=(0.0, 1.0, 4.0)):
    rows = []
    for n in sizes:
        for ratio in ratios:
            for r in heap_workload(n, ratio):
                rows.append({"n": n, "decrease_per_event": ratio, "structure": r["structure"],
                             "insert_ops": r["insert_ops"], "decrease_key_ops": r["decrease_key_ops"],
                             "extract_min_ops": r["extract_min_ops"], "total_ops": r["total_ops"],
                             "merge_ops": r["merge_ops"], "total_ms": r["total_ms"], "sorted_ok": r["sorted_ok"]})
    return rows


def knapsack_experiment(n=50, capacities=(100, 500, 1000, 5000)):
    import random
    rnd = random.Random(4)
    items = [(rnd.randint(1, 60), rnd.randint(1, 200), f"p{i}") for i in range(n)]
    rows = []
    for W in capacities:
        z, t_z = timed(zero_one_knapsack, items, W, repeat=1)
        f, t_f = timed(fractional_knapsack, items, W, repeat=1)
        rows.append({"n": n, "W": W, "dp_cells_nW": z["table_cells"], "dp_ms": round(t_z, 3),
                     "dp_value": z["total_value"], "fractional_ms": round(t_f, 4),
                     "fractional_value": f["total_value"],
                     "fractional_ge_dp": f["total_value"] >= z["total_value"]})
    return rows


def nqueens_experiment(ns=(4, 6, 8, 10, 12)):
    rows = []
    for n in ns:
        r, t = timed(solve_n_queens, n, repeat=1)
        rows.append({"n": n, "states_first_solution": r["states"], "n_factorial": math.factorial(n), "ms": round(t, 3)})
    return rows


EXPERIMENTS = [
    ("mst", "Prim vs Kruskal: sparse → dense", mst_experiment),
    ("floyd", "Floyd–Warshall: growth with zones", floyd_experiment),
    ("tsp", "TSP: Backtracking vs Branch & Bound", tsp_experiment),
    ("trees", "Red-Black Tree vs B-Tree", tree_experiment),
    ("heaps", "Binomial vs Fibonacci Heap", heap_experiment),
    ("knapsack", "Knapsack: DP cost grows with W", knapsack_experiment),
    ("nqueens", "N-Queen backtracking", nqueens_experiment),
]


def write_csv(name, rows):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", f"{name}.csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return path


if __name__ == "__main__":
    for name, title, fn in EXPERIMENTS:
        print(f"\n=== {title} ===")
        rows = fn()
        print(",".join(rows[0].keys()))
        for r in rows:
            print(",".join(str(v) for v in r.values()))
        print("saved ->", write_csv(name, rows))
