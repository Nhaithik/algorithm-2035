"""Graph algorithms: Prim (heap + matrix versions), Kruskal, Floyd-Warshall.

Every function returns operation counters so the framework can compare
algorithms by *work done*, not only by (noisy) wall-clock time.
"""
import heapq
from functools import cmp_to_key

INF = float("inf")


def _check_edges(n, edges):
    if n <= 0:
        raise ValueError("Number of nodes must be positive.")
    for u, v, _ in edges:
        if not (0 <= u < n and 0 <= v < n):
            raise ValueError(f"Edge endpoint out of range 0..{n - 1}: ({u}, {v}).")


def _result(chosen, n, ops):
    total = sum(w for _, _, w in chosen)
    return {
        "edges": [[u, v, w] for u, v, w in chosen],
        "total_cost": total,
        "connected": len(chosen) == n - 1,
        "components": n - len(chosen),
        "ops": ops,
    }


class _CountingHeap:
    """Binary min-heap that counts key comparisons (used only for operation counting)."""
    def __init__(self):
        self.a = []
        self.cmp = 0

    def push(self, x):
        a = self.a
        a.append(x)
        i = len(a) - 1
        while i > 0:
            p = (i - 1) // 2
            self.cmp += 1
            if a[i] < a[p]:
                a[i], a[p] = a[p], a[i]
                i = p
            else:
                break

    def pop(self):
        a = self.a
        top, last = a[0], a.pop()
        if a:
            a[0] = last
            i, n = 0, len(a)
            while True:
                l, r, s = 2 * i + 1, 2 * i + 2, i
                if l < n:
                    self.cmp += 1
                    if a[l] < a[s]:
                        s = l
                if r < n:
                    self.cmp += 1
                    if a[r] < a[s]:
                        s = r
                if s == i:
                    break
                a[i], a[s] = a[s], a[i]
                i = s
        return top


def prim_heap(n, edges, count_ops=False):
    """Prim with adjacency list + binary heap. O((V+E) log V) time, O(V+E) space.
    count_ops=True uses a hand-written heap that counts comparisons (slower)."""
    _check_edges(n, edges)
    adj = [[] for _ in range(n)]
    for u, v, w in edges:
        adj[u].append((w, v))
        adj[v].append((w, u))
    if count_ops:
        H = _CountingHeap()
        push, pop = H.push, H.pop
    else:
        H = None
        raw = []
        push = lambda x: heapq.heappush(raw, x)
        pop = lambda: heapq.heappop(raw)
    seen = [False] * n
    chosen = []
    scans = 0
    for start in range(n):          # restart => minimum spanning FOREST if disconnected
        if seen[start]:
            continue
        push((0, start, -1))
        while (H.a if count_ops else raw):
            w, u, parent = pop()
            if seen[u]:
                continue
            seen[u] = True
            if parent != -1:
                chosen.append((parent, u, w))
            for nw, v in adj[u]:
                scans += 1
                if not seen[v]:
                    push((nw, v, u))
    cmp = H.cmp if H else 0
    return _result(chosen, n, {"heap_comparisons": cmp, "edge_scans": scans, "total": cmp + scans})


def prim_matrix(n, edges):
    """Prim with adjacency matrix + linear scan. O(V^2) time and space,
    independent of E -- the classic choice for dense graphs."""
    _check_edges(n, edges)
    W = [[INF] * n for _ in range(n)]
    for u, v, w in edges:
        if w < W[u][v]:
            W[u][v] = W[v][u] = w
    in_tree = [False] * n
    key = [INF] * n
    parent = [-1] * n
    chosen = []
    ops = 0
    for _ in range(n):
        u, best = -1, INF
        for v in range(n):
            ops += 1
            if not in_tree[v] and (u == -1 or key[v] < best):
                u, best = v, key[v]
        in_tree[u] = True
        if parent[u] != -1:
            chosen.append((parent[u], u, key[u]))
        for v in range(n):
            ops += 1
            if not in_tree[v] and W[u][v] < key[v]:
                key[v] = W[u][v]
                parent[v] = u
    return _result(chosen, n, {"total": ops})


def kruskal_mst(n, edges, count_ops=False):
    """Kruskal: sort edges + union-find. O(E log E) time, O(V+E) space."""
    _check_edges(n, edges)
    comparisons = 0
    if count_ops:
        def cmp(a, b):
            nonlocal comparisons
            comparisons += 1
            return (a[2] > b[2]) - (a[2] < b[2])
        ordered = sorted(edges, key=cmp_to_key(cmp))
    else:
        ordered = sorted(edges, key=lambda e: e[2])
    parent = list(range(n))
    rank = [0] * n
    finds = 0

    def find(x):
        nonlocal finds
        finds += 1
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    chosen = []
    for u, v, w in ordered:
        ru, rv = find(u), find(v)
        if ru == rv:
            continue
        if rank[ru] < rank[rv]:
            ru, rv = rv, ru
        parent[rv] = ru
        if rank[ru] == rank[rv]:
            rank[ru] += 1
        chosen.append((u, v, w))
        if len(chosen) == n - 1:
            break
    return _result(chosen, n, {"sort_comparisons": comparisons, "find_calls": finds,
                               "total": comparisons + finds})


def floyd_warshall(matrix):
    """All-pairs shortest costs. O(V^3) time, O(V^2) space (dynamic programming).
    Returns (dist, info). info has ops and a negative_cycle flag."""
    n = len(matrix)
    if n == 0 or any(len(r) != n for r in matrix):
        raise ValueError("Matrix must be square and non-empty.")
    d = [row[:] for row in matrix]
    for i in range(n):
        d[i][i] = min(d[i][i], 0)
    ops = 0
    for k in range(n):
        dk = d[k]
        for i in range(n):
            dik = d[i][k]
            di = d[i]
            for j in range(n):
                ops += 1
                if dik + dk[j] < di[j]:
                    di[j] = dik + dk[j]
    neg = any(d[i][i] < 0 for i in range(n))
    return d, {"ops": ops, "negative_cycle": neg}
