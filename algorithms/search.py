"""Backtracking / Branch & Bound: N-Queen, Hamiltonian, TSP.
All searches report states generated / pruned and support a state cap so the
framework can demonstrate scalability limits instead of freezing."""

INF = float("inf")


# ------------------------------- N-Queen -----------------------------------
def solve_n_queens(n, count_all=False):
    """Backtracking with column + diagonal sets. Returns first solution
    (and total solution count if count_all). Worst-case exponential, O(n) aux space."""
    if not 1 <= n <= 14:
        raise ValueError("N-Queen demo supports n from 1 to 14.")
    cols, d1, d2 = set(), set(), set()
    place = [-1] * n
    states = 0
    first = None
    count = 0

    def bt(r):
        nonlocal states, first, count
        states += 1
        if r == n:
            count += 1
            if first is None:
                first = place[:]
            return not count_all          # stop at the first solution unless counting all
        for c in range(n):
            if c in cols or (r - c) in d1 or (r + c) in d2:
                continue
            place[r] = c
            cols.add(c); d1.add(r - c); d2.add(r + c)
            if bt(r + 1):
                return True
            cols.discard(c); d1.discard(r - c); d2.discard(r + c)
            place[r] = -1
        return False

    bt(0)
    board = None
    if first is not None:
        board = [["Q" if first[r] == c else "·" for c in range(n)] for r in range(n)]
    return {"found": first is not None, "columns": first or [], "board": board or [],
            "states": states, "solutions": count if count_all else (1 if first else 0)}


# ------------------------------ Hamiltonian --------------------------------
def hamiltonian(n, edges, cycle=True, max_states=2_000_000):
    """Backtracking search for a Hamiltonian cycle (or path) in an undirected graph."""
    if not 1 <= n <= 20:
        raise ValueError("Hamiltonian demo supports 1 to 20 vertices.")
    adj = [set() for _ in range(n)]
    for u, v in edges:
        if not (0 <= u < n and 0 <= v < n):
            raise ValueError(f"Edge endpoint out of range 0..{n - 1}: ({u}, {v}).")
        if u != v:
            adj[u].add(v); adj[v].add(u)
    states = 0
    aborted = False
    path = [0]
    used = [False] * n
    used[0] = True
    found = None

    def bt():
        nonlocal states, aborted, found
        states += 1
        if states > max_states:
            aborted = True
            return True
        if len(path) == n:
            if not cycle or 0 in adj[path[-1]]:
                found = path[:] + ([0] if cycle and n > 1 else [])
                return True
            return False
        for v in sorted(adj[path[-1]]):
            if not used[v]:
                used[v] = True; path.append(v)
                if bt():
                    return True
                path.pop(); used[v] = False
        return False

    if n == 1:
        return {"found": True, "route": [0], "states": 1, "aborted": False}
    starts = [0] if cycle else range(n)   # a cycle can start anywhere; a path cannot
    for s in starts:
        path[:] = [s]
        used[:] = [False] * n
        used[s] = True
        if bt():
            break
    return {"found": found is not None, "route": found, "states": states, "aborted": aborted}


# ---------------------------------- TSP ------------------------------------
def _check_matrix(matrix):
    n = len(matrix)
    if n < 2:
        raise ValueError("TSP needs at least two locations.")
    if any(len(r) != n for r in matrix):
        raise ValueError("Cost matrix must be square.")
    return n


def tsp_backtracking(matrix, max_states=5_000_000):
    """Plain backtracking: enumerate every feasible tour, no cost-based pruning.
    (n-1)! leaves in the worst case."""
    n = _check_matrix(matrix)
    best, best_route = INF, None
    states = 0
    aborted = False
    visited = [False] * n
    visited[0] = True
    route = [0]

    def dfs(u, cost):
        nonlocal best, best_route, states, aborted
        if aborted:
            return
        states += 1
        if states > max_states:
            aborted = True
            return
        if len(route) == n:
            back = matrix[u][0]
            if back != INF and cost + back < best:
                best, best_route = cost + back, route[:] + [0]
            return
        for v in range(n):
            if not visited[v] and matrix[u][v] != INF:
                visited[v] = True; route.append(v)
                dfs(v, cost + matrix[u][v])
                route.pop(); visited[v] = False

    dfs(0, 0)
    return {"route": best_route, "cost": best if best_route else None,
            "states_generated": states, "states_pruned": 0, "aborted": aborted}


def tsp_branch_and_bound(matrix, max_states=5_000_000):
    """Branch & Bound. Lower bound for a partial tour = cost so far + cheapest
    outgoing edge of the current city + cheapest outgoing edge of every
    unvisited city (each of those cities must still be left exactly once).
    Children are explored cheapest-first. Prunes when bound >= best known tour."""
    n = _check_matrix(matrix)
    min_out = []
    for i in range(n):
        m = min((matrix[i][j] for j in range(n) if j != i), default=INF)
        if m == INF:
            return {"route": None, "cost": None, "states_generated": 1,
                    "states_pruned": 0, "aborted": False}
        min_out.append(m)
    best, best_route = INF, None
    states = pruned = 0
    aborted = False
    visited = [False] * n
    visited[0] = True
    route = [0]
    remaining = sum(min_out) - min_out[0]

    def dfs(u, cost, remaining):
        nonlocal best, best_route, states, pruned, aborted
        if aborted:
            return
        states += 1
        if states > max_states:
            aborted = True
            return
        if cost + min_out[u] + remaining >= best:
            pruned += 1
            return
        if len(route) == n:
            back = matrix[u][0]
            if back != INF and cost + back < best:
                best, best_route = cost + back, route[:] + [0]
            return
        kids = sorted((matrix[u][v], v) for v in range(n)
                      if not visited[v] and matrix[u][v] != INF)
        for w, v in kids:
            visited[v] = True; route.append(v)
            dfs(v, cost + w, remaining - min_out[v])
            route.pop(); visited[v] = False

    dfs(0, 0, remaining)
    return {"route": best_route, "cost": best if best_route else None,
            "states_generated": states, "states_pruned": pruned, "aborted": aborted}


# ------------------------ Matrix Chain (bonus DP) ---------------------------
def matrix_chain(dims):
    """Optimal parenthesisation. dims has length n+1 for n matrices. O(n^3) time, O(n^2) space."""
    if len(dims) < 2 or any(d <= 0 for d in dims):
        raise ValueError("Give at least 2 positive dimensions (n matrices need n+1 numbers).")
    n = len(dims) - 1
    m = [[0] * n for _ in range(n)]
    s = [[0] * n for _ in range(n)]
    ops = 0
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            m[i][j] = INF
            for k in range(i, j):
                ops += 1
                c = m[i][k] + m[k + 1][j] + dims[i] * dims[k + 1] * dims[j + 1]
                if c < m[i][j]:
                    m[i][j], s[i][j] = c, k

    def paren(i, j):
        return f"A{i + 1}" if i == j else f"({paren(i, s[i][j])} × {paren(s[i][j] + 1, j)})"

    left = 0
    rows = dims[0]
    for k in range(1, n):
        left += rows * dims[k] * dims[k + 1]
    return {"min_multiplications": m[0][n - 1], "order": paren(0, n - 1),
            "left_to_right": left, "ops": ops}
