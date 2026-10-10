"""The integrated pipeline:
SCENARIO INPUT -> PROBLEM MODEL -> DECISION ENGINE -> EXECUTION -> ANALYSIS -> EXPLANATION
Every goal goes through the same five stages and returns the same result shape,
so the UI (and tests) treat all modules uniformly."""
import math

from algorithms.graphs import prim_heap, prim_matrix, kruskal_mst, floyd_warshall, INF
from algorithms.knapsack import fractional_knapsack, zero_one_knapsack, greedy_zero_one
from algorithms.search import (solve_n_queens, hamiltonian, tsp_backtracking,
                               tsp_branch_and_bound, matrix_chain)
from algorithms.structures import tree_workload, heap_workload
from . import complexity as C
from .decision_engine import decide
from .metrics import timed
from .problem_model import build_model, random_cost_matrix, GOALS


# --------------------------- block helpers (for UI) ---------------------------
def kv(title, d):
    return {"type": "kv", "title": title, "items": [(k, v) for k, v in d.items()]}


def table(title, headers, rows, highlight=None, note=None):
    if highlight is not None and 0 <= highlight < len(rows):
        rows = [list(r) for r in rows]
        rows[highlight][0] = "★ " + str(rows[highlight][0])
    return {"type": "table", "title": title, "headers": headers, "rows": rows,
            "highlight": highlight, "note": note}


def text(title, body):
    return {"type": "text", "title": title, "body": body}


def bars(title, labels, values, unit=""):
    return {"type": "bars", "title": title, "labels": labels, "values": values, "unit": unit}


def grid(title, rows):
    return {"type": "grid", "title": title, "rows": rows}


def _fmt(x):
    if x == INF:
        return "∞"
    if isinstance(x, float) and x.is_integer():
        return int(x)
    if isinstance(x, float):
        return round(x, 2)
    return x


# ------------------------------ executors --------------------------------
def _exec_connect(model, decision):
    n, edges = model["data"]["n"], model["data"]["edges"]
    pm, pm_ms = timed(prim_matrix, n, edges)
    ph, ph_ms = timed(prim_heap, n, edges)
    kr, kr_ms = timed(kruskal_mst, n, edges)
    ph_ops = prim_heap(n, edges, count_ops=True)["ops"]["total"]
    kr_ops = kruskal_mst(n, edges, count_ops=True)["ops"]["total"]
    rows = [["Prim (adjacency matrix)", pm["total_cost"], pm["ops"]["total"], round(pm_ms, 3)],
            ["Prim (binary heap)", ph["total_cost"], ph_ops, round(ph_ms, 3)],
            ["Kruskal", kr["total_cost"], kr_ops, round(kr_ms, 3)]]
    hi = next(i for i, r in enumerate(rows) if r[0] == decision["chosen"])
    agree = math.isclose(pm["total_cost"], kr["total_cost"]) and math.isclose(ph["total_cost"], kr["total_cost"])
    fastest_ops = min(rows, key=lambda r: r[2])[0]
    fastest_ms = min(rows, key=lambda r: r[3])[0]
    blocks = [
        table("Comparison (★ = chosen by the decision engine)", ["Algorithm", "Total cost", "Operations", "Time (ms)"], rows, hi,
              note=f"Fewest operations: {fastest_ops}. Fastest wall-clock here: {fastest_ms}. "
                   "Operations count comparisons/scans and follow the theory. Wall-clock can disagree in Python: "
                   "Kruskal's sort runs in native C while the matrix scan is interpreted, so no algorithm is 'always faster'."),
        kv("Correctness cross-check", {"All three give the same total cost": "yes ✓" if agree else "NO ✗ — bug"}),
    ]
    sol = kr
    blocks.append(kv("Solution", {"Total cable cost": _fmt(sol["total_cost"]),
                                  "Network connected": "yes" if sol["connected"] else f"no — {sol['components']} separate components (minimum spanning forest)",
                                  "Links chosen": len(sol["edges"])}))
    if len(sol["edges"]) <= 60:
        blocks.append(table("Chosen links", ["From", "To", "Cost"],
                            [[u, v, _fmt(w)] for u, v, w in sol["edges"]]))
    summary = f"{decision['chosen']} on V={n}, E={len(edges)}; MST cost {_fmt(sol['total_cost'])}"
    return blocks, summary


def _exec_all_pairs(model, decision):
    m = model["data"]["matrix"]
    n = len(m)
    (dist, info), ms = timed(floyd_warshall, m, repeat=2)
    blocks = []
    if info["negative_cycle"]:
        blocks.append(text("Warning", "A negative-cost cycle exists, so shortest costs are undefined for some pairs."))
    blocks.append(kv("Measured", {"Zones V": n, "Operations": f"{info['ops']:,}", "V³": f"{n ** 3:,}",
                                  "Time (ms)": round(ms, 3)}))
    if n <= 15:
        blocks.append(table("All-pairs cost matrix", [""] + [f"Z{j}" for j in range(n)],
                            [[f"Z{i}"] + [_fmt(x) for x in dist[i]] for i in range(n)]))
    else:
        blocks.append(text("Matrix", f"{n}×{n} matrix computed (too large to display)."))
    sizes = [s for s in (10, 20, 40, 80) if s <= 120]
    rows, prev = [], None
    for s in sizes:
        mm = random_cost_matrix(s, 0.5, 1)
        (_, inf2), t = timed(floyd_warshall, mm, repeat=1)
        rows.append([s, f"{inf2['ops']:,}", f"{s ** 3:,}", round(t, 3), "—" if prev is None else f"{t / prev:.1f}×"])
        prev = t
    blocks.append(table("Effect of increasing the number of zones", ["Zones V", "Operations", "V³", "Time (ms)", "Time vs previous (V doubles)"], rows,
                        note="Doubling V multiplies the work by about 8 — cubic growth."))
    blocks.append(bars("Operations by number of zones", [str(r[0]) for r in rows], [int(r[1].replace(",", "")) for r in rows]))
    return blocks, f"Floyd–Warshall on {n} zones, {info['ops']:,} operations"


def _exec_budget(model, decision):
    items, cap = model["data"]["items"], model["data"]["capacity"]
    blocks = []
    frac = fractional_knapsack(items, cap)
    zero, gr = None, None
    err = None
    try:
        zero = zero_one_knapsack(items, cap)
        gr = greedy_zero_one(items, cap)
    except ValueError as e:
        err = str(e)
    rows = [["Fractional Knapsack", frac["total_value"], "Projects may be split",
             ", ".join(f"{s['name']} ×{s['fraction']}" for s in frac["selected"])]]
    if zero:
        rows.append(["0/1 Knapsack (DP)", zero["total_value"], "Whole projects only",
                     ", ".join(s["name"] for s in zero["selected"]) or "—"])
        rows.append(["Ratio-greedy on whole projects", gr["total_value"], "Whole projects only (heuristic)",
                     ", ".join(gr["selected"]) or "—"])
    hi = 0 if model["facts"]["divisible"] else 1
    blocks.append(table("Both formulations on the same data (★ = chosen)", ["Method", "Benefit", "Meaning", "Selected"], rows, hi if zero or hi == 0 else 0))
    if err:
        blocks.append(text("0/1 DP not run", err))
    if zero:
        gap = round(frac["total_value"] - zero["total_value"], 4)
        blocks.append(kv("What the comparison shows", {
            "Fractional is an upper bound on 0/1": f"{frac['total_value']} ≥ {zero['total_value']} (gap {gap})",
            "Ratio-greedy loses to the DP by": round(zero["total_value"] - gr["total_value"], 4),
            "0/1 DP table cells (n·W)": f"{zero['table_cells']:,}"}))
    if model["facts"]["divisible"]:
        summary = f"{decision['chosen']}; benefit {frac['total_value']}"
    elif zero:
        summary = f"{decision['chosen']}; benefit {zero['total_value']}"
    else:
        summary = f"{decision['chosen']} could not run: {err}"
    return blocks, summary


def _exec_records(model, decision):
    d = model["data"]
    rows = tree_workload(d["n"], d["ratio"], d["t"])
    rb, bt = rows
    blocks = [table("Measured on the same workload", ["Metric", rb["structure"], bt["structure"]], [
        ["Tree height (levels)", rb["height"], bt["height"]],
        ["Node visits during search (≈ page reads)", f"{rb['search_node_visits']:,}", f"{bt['search_node_visits']:,}"],
        ["Key comparisons during search", f"{rb['search_comparisons']:,}", f"{bt['search_comparisons']:,}"],
        ["Rebalancing work during inserts", f"{rb['rebalance_ops']:,} {rb['rebalance_kind']}", f"{bt['rebalance_ops']:,} {bt['rebalance_kind']}"],
        ["Insert time (ms)", rb["insert_ms"], bt["insert_ms"]],
        ["Search time (ms)", rb["search_ms"], bt["search_ms"]],
        ["Structure valid after workload", "yes ✓" if rb["valid"] else "NO", "yes ✓" if bt["valid"] else "NO"],
    ], note="Both are real implementations. On disk, cost ≈ node visits (each visit = one page read); in RAM, time and comparisons matter more.")]
    blocks.append(bars("Node visits during search", ["Red-Black", "B-Tree"], [rb["search_node_visits"], bt["search_node_visits"]]))
    return blocks, f"{decision['chosen']}; RB height {rb['height']} vs B-Tree height {bt['height']}"


def _exec_priorities(model, decision):
    d = model["data"]
    bi, fi = heap_workload(d["n"], d["ratio"])
    blocks = [table("Measured elementary steps on the same event workload", ["Phase", "Binomial Heap", "Fibonacci Heap"], [
        ["Insert all events", f"{bi['insert_ops']:,}", f"{fi['insert_ops']:,}"],
        [f"{bi['decrease_keys']:,} priority escalations (decrease-key)", f"{bi['decrease_key_ops']:,}", f"{fi['decrease_key_ops']:,}"],
        ["Serve all events (extract-min)", f"{bi['extract_min_ops']:,}", f"{fi['extract_min_ops']:,}"],
        ["TOTAL", f"{bi['total_ops']:,}", f"{fi['total_ops']:,}"],
        ["Merge two heaps of n/2 events", f"{bi['merge_ops']:,}", f"{fi['merge_ops']:,}"],
        ["Wall-clock total (ms)", bi["total_ms"], fi["total_ms"]],
        ["Events served in correct order", "yes ✓" if bi["sorted_ok"] else "NO", "yes ✓" if fi["sorted_ok"] else "NO"],
    ], note="Steps = root visits + links + cuts + swaps. Step counts ignore constant factors (pointer-heavy Fibonacci nodes), so compare wall-clock too; Fibonacci's O(1) insert/merge shows in steps, its decrease-key advantage grows with the escalation ratio.")]
    blocks.append(bars("Total elementary steps", ["Binomial", "Fibonacci"], [bi["total_ops"], fi["total_ops"]]))
    return blocks, f"{decision['chosen']}; steps {bi['total_ops']:,} (binomial) vs {fi['total_ops']:,} (Fibonacci)"


def _exec_placement(model, decision):
    n = model["data"]["n"]
    one, ms = timed(solve_n_queens, n, repeat=1)
    blocks = []
    if one["found"]:
        blocks.append(grid(f"Sensor layout on {n}×{n} grid (Q = sensor)", one["board"]))
    else:
        blocks.append(text("Result", f"No conflict-free placement exists for n = {n} (N-Queen has no solution for n = 2 and 3)."))
    rows = [["Backtracking (first solution)", f"{one['states']:,}", round(ms, 3)]]
    if n <= 10:
        allr = solve_n_queens(n, count_all=True)
        rows.append(["Backtracking (count ALL solutions)", f"{allr['states']:,}", "—"])
        blocks.append(kv("Solutions", {"Distinct conflict-free layouts": allr["solutions"]}))
    rows.append(["Brute force: n! column permutations", f"{math.factorial(n):,}", "not run"])
    blocks.append(table("Search effort", ["Method", "States", "Time (ms)"], rows))
    return blocks, f"Backtracking, n={n}, {one['states']:,} states"


def _tour_rows(m, n):
    rows, results = [], {}
    if n <= 10:
        bt, bt_ms = timed(tsp_backtracking, m, repeat=1)
        results["bt"] = (bt, bt_ms)
    bb, bb_ms = timed(tsp_branch_and_bound, m, repeat=1)
    results["bb"] = (bb, bb_ms)
    return results


def _exec_tour(model, decision):
    m = model["data"]["matrix"]
    n = len(m)
    blocks = []
    if decision["chosen"].startswith("REFUSED"):
        blocks.append(text("Scalability limit reached",
                           f"Exact TSP for {n} sites would need to consider up to {math.factorial(n - 1):,} tours. "
                           "The framework refuses to run it. Reduce the number of sites, or use an approximation (outside the prescribed syllabus)."))
        return blocks, f"Refused exact TSP for n={n}"
    res = _tour_rows(m, n)
    rows = []
    if "bt" in res:
        r, t = res["bt"]
        rows.append(["Backtracking", _fmt(r["cost"]) if r["cost"] is not None else "none", f"{r['states_generated']:,}", 0, round(t, 3)])
    r, t = res["bb"]
    rows.append(["Branch & Bound", _fmt(r["cost"]) if r["cost"] is not None else "none", f"{r['states_generated']:,}", f"{r['states_pruned']:,}", round(t, 3)])
    hi = 0 if (decision["chosen"] == "TSP Backtracking" and "bt" in res) else len(rows) - 1
    blocks.append(table("Exact search comparison (★ = chosen)", ["Method", "Best tour cost", "States generated", "Subtrees pruned", "Time (ms)"], rows, hi))
    chosen_res = res["bt"][0] if (decision["chosen"] == "TSP Backtracking" and "bt" in res) else res["bb"][0]
    if chosen_res["route"]:
        blocks.append(kv("Optimal tour", {"Route": " → ".join(map(str, chosen_res["route"])), "Total cost": _fmt(chosen_res["cost"])}))
        if "bt" in res:
            same = res["bt"][0]["cost"] == res["bb"][0]["cost"]
            blocks.append(kv("Correctness cross-check", {"Backtracking and Branch & Bound agree": "yes ✓" if same else "NO ✗"}))
    else:
        blocks.append(text("Result", "No tour exists with the given links."))
    blocks.append(text("Why exact TSP stops scaling", f"There are (n-1)! = {math.factorial(n - 1):,} tours for n = {n}. Adding one site multiplies that by n. See the Complexity page for measured growth."))
    return blocks, f"{decision['chosen']}, n={n}, cost {_fmt(chosen_res['cost']) if chosen_res['route'] else 'none'}"


def _exec_hamiltonian(model, decision):
    d = model["data"]
    r = hamiltonian(d["n"], d["edges"], d["cycle"])
    kind = "cycle" if d["cycle"] else "path"
    blocks = []
    if r["aborted"]:
        blocks.append(text("Search aborted", "State limit reached — this is the NP-Complete behaviour: the search space blew up."))
    elif r["found"]:
        blocks.append(kv("Feasible ✓", {f"Hamiltonian {kind}": " → ".join(map(str, r["route"])), "States explored": r["states"]}))
    else:
        blocks.append(kv("Not feasible ✗", {"Result": f"No Hamiltonian {kind} exists; all branches exhausted.", "States explored": r["states"]}))
    return blocks, f"Hamiltonian {kind}: {'found' if r['found'] else 'none'}, {r['states']:,} states"


def _exec_matrix_chain(model, decision):
    r = matrix_chain(model["data"]["dims"])
    saving = r["left_to_right"] - r["min_multiplications"]
    blocks = [kv("Result", {"Best evaluation order": r["order"], "Scalar multiplications (optimal)": f"{r['min_multiplications']:,}",
                            "Left-to-right order": f"{r['left_to_right']:,}",
                            "Work saved": f"{saving:,} ({(saving / r['left_to_right'] * 100) if r['left_to_right'] else 0:.0f}%)"})]
    return blocks, f"Optimal {r['min_multiplications']:,} vs left-to-right {r['left_to_right']:,}"


EXECUTORS = {"connect_sites": _exec_connect, "all_pairs": _exec_all_pairs, "budget_projects": _exec_budget,
             "city_records": _exec_records, "maintenance_priorities": _exec_priorities,
             "sensor_placement": _exec_placement, "inspection_tour": _exec_tour,
             "route_feasibility": _exec_hamiltonian, "matrix_chain": _exec_matrix_chain}


def run(goal, params):
    """Run the full pipeline. Returns a dict with model, decision, blocks, summary."""
    if goal not in EXECUTORS:
        raise ValueError(f"Unknown goal '{goal}'.")
    model = build_model(goal, params)               # stage 2: problem modelling
    decision = decide(goal, model["facts"])          # stage 3: adaptive decision
    blocks, summary = EXECUTORS[goal](model, decision)   # stages 4-5: execute + analyse
    return {"goal": goal, "title": GOALS[goal], "model": model, "decision": decision,
            "blocks": blocks, "summary": summary}
