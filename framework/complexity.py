"""Complexity knowledge base: time/space, correctness argument, and exact
NP-class statements (decision vs optimisation). Used by the decision engine,
the pages and the report."""

ALGORITHMS = {
    "Prim (binary heap)": {
        "time": "O((V+E) log V)", "space": "O(V+E)",
        "correctness": "Cut property: the cheapest edge leaving the current tree is in some MST.",
        "class": "Polynomial"},
    "Prim (adjacency matrix)": {
        "time": "O(V²)", "space": "O(V²)",
        "correctness": "Same cut-property argument; array scan replaces the heap.",
        "class": "Polynomial"},
    "Kruskal": {
        "time": "O(E log E) = O(E log V)", "space": "O(V+E)",
        "correctness": "Cycle property: the cheapest edge not forming a cycle is safe (union-find detects cycles).",
        "class": "Polynomial"},
    "Floyd–Warshall": {
        "time": "O(V³)", "space": "O(V²)",
        "correctness": "DP invariant: after phase k, d[i][j] is the cheapest i→j path using only intermediate vertices 0..k.",
        "class": "Polynomial"},
    "Fractional Knapsack (greedy)": {
        "time": "O(n log n)", "space": "O(n)",
        "correctness": "Exchange argument: swapping any weight for a higher value/weight weight never lowers value.",
        "class": "Polynomial"},
    "0/1 Knapsack (DP)": {
        "time": "O(nW)", "space": "O(nW)",
        "correctness": "Optimal substructure: best(i,c) = max(best(i-1,c), v_i + best(i-1,c-w_i)).",
        "class": "Pseudo-polynomial (W is a numeric value, not input length)"},
    "Red-Black Tree": {
        "time": "search/insert O(log n); ≤2 rotations per insert", "space": "O(n)",
        "correctness": "Colour rules bound height by 2·log₂(n+1).",
        "class": "Polynomial"},
    "B-Tree": {
        "time": "search/insert O(t·log_t n) CPU, O(log_t n) node accesses", "space": "O(n)",
        "correctness": "All leaves at equal depth; every non-root node holds t-1..2t-1 keys.",
        "class": "Polynomial"},
    "Binomial Heap": {
        "time": "insert, merge, extract-min, decrease-key: O(log n) worst case", "space": "O(n)",
        "correctness": "A heap of n items has at most ⌊log₂ n⌋+1 binomial trees, one per set bit of n.",
        "class": "Polynomial"},
    "Fibonacci Heap": {
        "time": "insert, merge, decrease-key: O(1) amortized; extract-min: O(log n) amortized", "space": "O(n)",
        "correctness": "Potential method; node degree ≤ O(log n) because a degree-k subtree has ≥ F(k+2) nodes.",
        "class": "Polynomial (amortized)"},
    "N-Queen backtracking": {
        "time": "exponential worst case (O(n!) placements bound)", "space": "O(n) auxiliary",
        "correctness": "Exhaustive DFS over rows; a partial placement is abandoned only when it already has a conflict.",
        "class": "Search problem; standard N-Queen is NOT labelled NP-Complete"},
    "Hamiltonian backtracking": {
        "time": "O(n!) worst case", "space": "O(n)",
        "correctness": "Exhaustive DFS over simple paths; infeasible only when no unvisited neighbour exists.",
        "class": "Decision problem is NP-Complete"},
    "TSP backtracking": {
        "time": "O((n-1)!)", "space": "O(n)",
        "correctness": "Enumerates every Hamiltonian tour from a fixed start; keeps the cheapest.",
        "class": "Optimisation NP-Hard; decision NP-Complete"},
    "TSP branch & bound": {
        "time": "O((n-1)!) worst case; far fewer states in practice", "space": "O(n)",
        "correctness": "Prunes a subtree only when its lower bound ≥ best known tour, so the optimum is never discarded.",
        "class": "Optimisation NP-Hard; decision NP-Complete"},
    "Matrix Chain (DP)": {
        "time": "O(n³)", "space": "O(n²)",
        "correctness": "Optimal substructure on split point k.",
        "class": "Polynomial"},
}

NP_STATEMENTS = {
    "TSP": ("Optimisation TSP (minimum-cost tour) is NP-Hard. Decision TSP "
            "('is there a tour of cost ≤ K?') is NP-Complete."),
    "Hamiltonian": ("Deciding whether a Hamiltonian cycle/path exists is NP-Complete; "
                    "no polynomial algorithm is known."),
    "0/1 Knapsack": ("The DP runs in O(nW): pseudo-polynomial, polynomial in the numeric value W but "
                     "exponential in the number of bits of W. The decision version is NP-Complete."),
    "Fractional Knapsack": "Solvable in polynomial time O(n log n) by a greedy algorithm — not NP-Hard.",
    "N-Queen": ("Backtracking is exponential in the worst case, but standard N-Queen is not "
                "automatically classified NP-Complete (solutions can be constructed directly for every n ≥ 4)."),
    "MST": "Polynomial-time solvable (Prim/Kruskal).",
    "All-pairs": "Polynomial-time solvable, O(V³).",
    "Matrix Chain": "Polynomial-time solvable by DP, O(n³).",
    "Trees/Heaps": "Data-structure operations are polynomial (logarithmic) — no NP aspect.",
}

# Gaps in classic algorithms that motivated newer algorithms (rubric: experimental analysis).
GAPS = [
    ("Binomial Heap", "decrease-key and insert cost O(log n) each, costly for graph algorithms that call decrease-key ~E times.",
     "Fibonacci Heap: O(1) amortized decrease-key. Prim/Dijkstra with it run in O(E + V log V)."),
    ("Binary search trees", "an unbalanced BST degrades to O(n) height on sorted input.",
     "Red-Black Tree / AVL Tree keep height O(log n); B-Tree extends this to wide nodes for disk."),
    ("Red-Black Tree on disk", "binary nodes mean ~log₂ n page reads per search.",
     "B-Tree: branching factor t cuts page reads to log_t n."),
    ("Prim vs Kruskal", "Kruskal's sort costs E log E; Prim-with-array costs V² whatever E is. Neither dominates.",
     "Borůvka and Fredman–Tarjan (Prim + Fibonacci heap) give near-linear-time variants; the density rule here picks between the classic two."),
    ("Floyd–Warshall", "O(V³) is wasteful on sparse graphs.",
     "Johnson's algorithm: O(V² log V + VE) for sparse graphs."),
    ("Greedy for 0/1 Knapsack", "ratio-greedy is not optimal for indivisible items (see the demo).",
     "Dynamic programming gives exact answers in O(nW); an FPTAS gives (1-ε) approximation when W is huge."),
    ("Exact TSP backtracking", "(n-1)! states — unusable beyond ~12–15 cities.",
     "Branch & Bound prunes; Held–Karp DP needs O(n²·2ⁿ); Christofides gives a 1.5-approximation for metric TSP."),
    ("Brute-force N-Queen", "n! or nⁿ placements.",
     "Backtracking prunes conflicts early, reducing the search to a small fraction of the permutations."),
]


def classify(problem):
    return NP_STATEMENTS.get(problem, "No NP-class statement for this problem.")
