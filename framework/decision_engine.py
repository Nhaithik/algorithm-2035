"""Stage 3 of the pipeline: ADAPTIVE DECISION ENGINE.
Rules use PROBLEM PROPERTIES (density, divisibility, storage, decrease-key
ratio, number of sites ...) -- never an algorithm name supplied by the user.
Every decision returns: problem type, chosen technique, rejected alternatives
with reasons, the rule that fired, expected complexity and NP class."""
import math

from . import complexity as C

# Tunable thresholds (documented in docs/DAA_analysis.md)
HEAP_N_THRESHOLD = 1000          # below this, Fibonacci's big constants do not pay off
TSP_BACKTRACK_MAX = 8            # (n-1)! <= 5040 leaves: plain backtracking is instant
TSP_EXACT_MAX = 13               # above this, exact search is refused in the demo


def _d(problem_type, chosen, rejected, rule, facts, complexity, np_note):
    return {"problem_type": problem_type, "chosen": chosen, "rejected": rejected,
            "rule": rule, "facts": facts, "complexity": complexity, "np_note": np_note}


def decide(goal, f):
    if goal == "connect_sites":
        V, E = f["V"], f["E"]
        prim_cost = 2 * V * V                       # matrix Prim: two V-length scans per iteration
        kr_cost = E * math.log2(max(E, 2))          # Kruskal: sort E edges (union-find adds O(E α))
        facts = {"V": V, "E": E, "density": f"{f['density']:.1%}",
                 "Prim (matrix) modelled cost 2·V²": f"{prim_cost:,.0f}",
                 "Kruskal modelled cost E·log₂E": f"{kr_cost:,.0f}"}
        heap_note = ("Prim (binary heap) is also run as a reference: it is O((V+E) log V), the same order as Kruskal, "
                     "so the rule contrasts the two classic extremes — array-Prim (cost independent of E) versus sort-based Kruskal.")
        rule = "Choose Prim (matrix) if 2·V² < E·log₂E, otherwise Kruskal. " + heap_note
        kind = "dense" if f["density"] >= 0.5 else "sparse" if f["density"] < 0.2 else "medium-density"
        if prim_cost < kr_cost:
            return _d(f"Minimum Spanning Tree ({kind} graph)", "Prim (adjacency matrix)",
                      [{"name": "Kruskal", "reason": f"Sorting E = {E:,} edges costs ≈ {kr_cost:,.0f} steps, more than Prim-matrix's ≈ {prim_cost:,.0f}."}],
                      rule, facts, C.ALGORITHMS["Prim (adjacency matrix)"]["time"], C.classify("MST"))
        return _d(f"Minimum Spanning Tree ({kind} graph)", "Kruskal",
                  [{"name": "Prim (adjacency matrix)", "reason": f"Scanning a V×V matrix costs ≈ {prim_cost:,.0f} steps regardless of edges, more than Kruskal's ≈ {kr_cost:,.0f} on only {E:,} edges."}],
                  rule, facts, C.ALGORITHMS["Kruskal"]["time"], C.classify("MST"))

    if goal == "all_pairs":
        V = f["V"]
        return _d("All-pairs shortest costs", "Floyd–Warshall",
                  [{"name": "Repeated single-source shortest paths (not in syllabus)",
                    "reason": "Needs V separate runs and cannot handle negative links without extra machinery; Floyd–Warshall gives all V² answers in one O(V³) DP and tolerates negative links."}],
                  "Rule: the request needs costs for EVERY pair → all-pairs DP.",
                  {"V": V, "ops expected V³": f"{V ** 3:,}", "negative links": f["negative_edges"]},
                  C.ALGORITHMS["Floyd–Warshall"]["time"] + " time, " + C.ALGORITHMS["Floyd–Warshall"]["space"] + " space",
                  C.classify("All-pairs"))

    if goal == "budget_projects":
        if f["divisible"]:
            return _d("Fractional Knapsack (divisible budget)", "Fractional Knapsack (greedy by benefit/cost)",
                      [{"name": "0/1 Knapsack DP", "reason": "Projects can be part-funded, so the greedy exchange argument gives the optimum in O(n log n); DP is unnecessary work."}],
                      "Rule: items divisible → greedy by ratio.", {"divisible": True, "n": f["n"]},
                      C.ALGORITHMS["Fractional Knapsack (greedy)"]["time"], C.classify("Fractional Knapsack"))
        warn = ""
        if not f["integral"]:
            warn = " (Costs/budget are not whole numbers: scale them to integers for the DP.)"
        return _d("0/1 Knapsack (indivisible projects)", "0/1 Knapsack (dynamic programming)",
                  [{"name": "Fractional Knapsack", "reason": "It may fund half a project, which is infeasible; and ratio-greedy on whole projects is not optimal (see demo)."}],
                  "Rule: items indivisible → DP over (item, budget)." + warn,
                  {"divisible": False, "n": f["n"], "W": f["W"], "DP cells n·W": f"{int(f['n'] * f['W']):,}"},
                  C.ALGORITHMS["0/1 Knapsack (DP)"]["time"] + " time (pseudo-polynomial)", C.classify("0/1 Knapsack"))

    if goal == "city_records":
        facts = {"n": f["n"], "storage": f["storage"], "lookups per insert": f["search_ratio"]}
        if f["storage"] == "disk":
            return _d("Ordered searchable data (paged storage)", "B-Tree",
                      [{"name": "Red-Black Tree", "reason": "Binary nodes → about log₂ n page reads per lookup; a B-Tree reads only log_t n pages."}],
                      "Rule: data on disk/paged storage → minimise node (page) accesses → B-Tree.",
                      facts, C.ALGORITHMS["B-Tree"]["time"], C.classify("Trees/Heaps"))
        return _d("Ordered searchable data (in-memory)", "Red-Black Tree",
                  [{"name": "B-Tree", "reason": "In RAM there are no page reads to save; wide nodes just add in-node scanning and split copying."}],
                  "Rule: data in RAM → simple binary balanced tree, O(1) rotations per insert → Red-Black Tree.",
                  facts, C.ALGORITHMS["Red-Black Tree"]["time"], C.classify("Trees/Heaps"))

    if goal == "maintenance_priorities":
        r, n = f["decrease_ratio"], f["n"]
        facts = {"n": n, "decrease-keys per event": r, "n threshold": HEAP_N_THRESHOLD}
        if r >= 1 and n >= HEAP_N_THRESHOLD:
            return _d("Mergeable priority queue (decrease-key heavy)", "Fibonacci Heap",
                      [{"name": "Binomial Heap", "reason": f"With {r:g} decrease-keys per event, O(log n) each adds up; Fibonacci does them in O(1) amortized."}],
                      f"Rule: decrease-key ratio ≥ 1 and n ≥ {HEAP_N_THRESHOLD} → Fibonacci Heap.",
                      facts, C.ALGORITHMS["Fibonacci Heap"]["time"], C.classify("Trees/Heaps"))
        return _d("Mergeable priority queue (balanced / small workload)", "Binomial Heap",
                  [{"name": "Fibonacci Heap", "reason": "Its O(1) decrease-key/insert only pay off when decrease-keys are frequent and n is large; here its larger constants (pointer-heavy nodes, cascading cuts) and amortized-only bounds are not worth it."}],
                  f"Rule: decrease-key ratio < 1 or n < {HEAP_N_THRESHOLD} → Binomial Heap (worst-case O(log n) guarantees).",
                  facts, C.ALGORITHMS["Binomial Heap"]["time"], C.classify("Trees/Heaps"))

    if goal == "sensor_placement":
        n = f["n"]
        return _d("Conflict-free placement (N-Queen model)", "Backtracking (row-by-row with conflict sets)",
                  [{"name": "Brute-force placement", "reason": f"Tries up to n^n = {n ** n:,} placements (n! = {math.factorial(n):,} if one per column); backtracking abandons a branch the moment a conflict appears."}],
                  "Rule: pairwise row/column/diagonal conflicts on a grid → N-Queen backtracking.",
                  {"n": n}, C.ALGORITHMS["N-Queen backtracking"]["time"], C.classify("N-Queen"))

    if goal == "inspection_tour":
        n = f["n"]
        facts = {"sites n": n, "(n-1)! tours": f"{math.factorial(n - 1):,}"}
        if n <= TSP_BACKTRACK_MAX:
            return _d("Minimum complete tour (small TSP)", "TSP Backtracking",
                      [{"name": "Branch & Bound", "reason": f"Only {math.factorial(n - 1):,} tours — enumeration is instant; bounding code adds no real benefit."}],
                      f"Rule: n ≤ {TSP_BACKTRACK_MAX} → plain backtracking.",
                      facts, C.ALGORITHMS["TSP backtracking"]["time"], C.classify("TSP"))
        if n <= TSP_EXACT_MAX:
            return _d("Minimum complete tour (medium TSP)", "TSP Branch & Bound",
                      [{"name": "TSP Backtracking", "reason": f"Enumerating {math.factorial(n - 1):,} tours is wasteful; the lower bound prunes most of them."}],
                      f"Rule: {TSP_BACKTRACK_MAX} < n ≤ {TSP_EXACT_MAX} → Branch & Bound.",
                      facts, C.ALGORITHMS["TSP branch & bound"]["time"], C.classify("TSP"))
        return _d("Minimum complete tour (too large for exact search)", "REFUSED — exact search not run",
                  [{"name": "TSP Backtracking / Branch & Bound", "reason": f"n = {n} → up to {math.factorial(n - 1):,} tours. Optimisation TSP is NP-Hard, so the framework limits exact search to n ≤ {TSP_EXACT_MAX}."}],
                  f"Rule: n > {TSP_EXACT_MAX} → exact methods infeasible; use fewer sites or an approximation/heuristic (outside this syllabus).",
                  facts, C.ALGORITHMS["TSP branch & bound"]["time"], C.classify("TSP"))

    if goal == "route_feasibility":
        n = f["n"]
        return _d("Hamiltonian feasibility", "Hamiltonian backtracking",
                  [{"name": "TSP", "reason": "There are no costs to minimise — only feasibility — so the optimisation machinery is unnecessary."}],
                  "Rule: visit-each-vertex-once feasibility with no cost objective → Hamiltonian backtracking.",
                  {"V": n, "E": f["E"], "cycle required": f["cycle"]},
                  C.ALGORITHMS["Hamiltonian backtracking"]["time"], C.classify("Hamiltonian"))

    if goal == "matrix_chain":
        return _d("Optimal matrix evaluation order", "Matrix Chain DP",
                  [{"name": "Left-to-right evaluation", "reason": "Ignores dimensions; can cost far more scalar multiplications."}],
                  "Rule: associative product with different dimensions → DP on split point.",
                  {"matrices": f["n"]}, C.ALGORITHMS["Matrix Chain (DP)"]["time"], C.classify("Matrix Chain"))

    return _d("Unknown", "Review input model", [], "No rule matches.", {}, "—", "—")
