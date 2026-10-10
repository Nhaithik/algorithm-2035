"""Stage 2 of the pipeline: PROBLEM MODELLING.
Turns raw user input (strings from a form) into a typed model:
entities, constraints, objective, size parameters and the 'facts' the decision
engine uses.  Also holds generators for random instances."""
import math
import random

INF = float("inf")

GOALS = {
    "connect_sites": "Connect sites with minimum total cable cost",
    "all_pairs": "Know the cheapest cost between every pair of zones",
    "budget_projects": "Choose projects within a budget",
    "city_records": "Store and search city sensor records",
    "maintenance_priorities": "Manage changing maintenance priorities",
    "sensor_placement": "Place mutually-interfering sensors without conflict",
    "inspection_tour": "Find the cheapest inspection tour through all sites",
    "route_feasibility": "Check whether a route can visit every zone exactly once",
    "matrix_chain": "Order a chain of matrix computations to minimise work",
}


# ------------------------------- generators -------------------------------
def random_graph(n, density, seed=7):
    """Connected random weighted graph. Edge (u,v) present with probability `density`."""
    rnd = random.Random(seed)
    edges, present = [], set()
    for u in range(n):
        for v in range(u + 1, n):
            if rnd.random() < density:
                edges.append((u, v, rnd.randint(1, 1000)))
                present.add((u, v))
    for u in range(n - 1):                      # guarantee connectivity
        if (u, u + 1) not in present:
            edges.append((u, u + 1, rnd.randint(1, 1000)))
    return edges


def random_cost_matrix(n, density=0.5, seed=3):
    rnd = random.Random(seed)
    m = [[0 if i == j else INF for j in range(n)] for i in range(n)]
    for i in range(n):
        m[i][(i + 1) % n] = rnd.randint(1, 50)   # ring => strongly connected
        for j in range(n):
            if i != j and rnd.random() < density:
                m[i][j] = min(m[i][j], rnd.randint(1, 50))
    return m


def random_tour_matrix(n, seed=5):
    """Symmetric Euclidean-style distances between n random sites."""
    rnd = random.Random(seed)
    pts = [(rnd.randint(0, 100), rnd.randint(0, 100)) for _ in range(n)]
    return [[round(math.dist(a, b)) for b in pts] for a in pts]


# --------------------------------- parsers --------------------------------
def _int(params, key, default, lo, hi):
    try:
        v = int(float(str(params.get(key, default)).strip() or default))
    except ValueError:
        raise ValueError(f"'{key}' must be a number.")
    if not lo <= v <= hi:
        raise ValueError(f"'{key}' must be between {lo} and {hi}.")
    return v


def _float(params, key, default, lo, hi):
    try:
        v = float(str(params.get(key, default)).strip() or default)
    except ValueError:
        raise ValueError(f"'{key}' must be a number.")
    if not lo <= v <= hi:
        raise ValueError(f"'{key}' must be between {lo} and {hi}.")
    return v


def parse_edges(raw, nodes, weighted=True):
    edges = []
    for part in raw.replace("\n", ";").split(";"):
        if not part.strip():
            continue
        bits = [x.strip() for x in part.replace("-", ",").split(",")] if not weighted \
            else [x.strip() for x in part.split(",")]
        need = 3 if weighted else 2
        if len(bits) != need:
            raise ValueError("Edges look like '0,1,4;1,2,3'" if weighted else "Edges look like '0-1;1-2'")
        a, b = int(bits[0]), int(bits[1])
        if not (0 <= a < nodes and 0 <= b < nodes):
            raise ValueError(f"Edge endpoints must be between 0 and {nodes - 1}.")
        if a == b:
            continue
        edges.append((a, b, float(bits[2])) if weighted else (a, b))
    return edges


def parse_matrix(raw):
    rows = []
    for row in raw.replace("\n", ";").strip().split(";"):
        if not row.strip():
            continue
        rows.append([INF if x.strip().upper() in ("INF", "INFINITY", "∞") else float(x)
                     for x in row.split(",")])
    if not rows or any(len(r) != len(rows) for r in rows):
        raise ValueError("Enter a square matrix: rows separated by ';', values by ','. Use INF for no link.")
    return rows


def parse_items(raw):
    items = []
    for part in raw.replace("\n", ";").split(";"):
        if not part.strip():
            continue
        bits = [x.strip() for x in part.split(",", 2)]
        if len(bits) != 3:
            raise ValueError("Each project must be 'cost,benefit,name'.")
        items.append((float(bits[0]), float(bits[1]), bits[2]))
    if not items:
        raise ValueError("Enter at least one project.")
    return items


# ------------------------------- the models --------------------------------
def build_model(goal, p):
    """Return model dict: entities, constraints, objective, size, data, facts."""
    if goal == "connect_sites":
        n = _int(p, "nodes", 8, 2, 400)
        raw = (p.get("edges") or "").strip()
        if raw:
            edges = parse_edges(raw, n)
            src = "user-entered"
        else:
            density = _float(p, "density", 0.3, 0.0, 1.0)
            edges = random_graph(n, density, _int(p, "seed", 7, 0, 10**6))
            src = f"randomly generated (target density {density:.0%})"
        e = len(edges)
        full = n * (n - 1) / 2
        return {"goal": goal, "data": {"n": n, "edges": edges},
                "entities": f"{n} city sites (vertices); {e} candidate cable links (weighted edges) — {src}",
                "constraints": "Every site must be reachable; links are undirected.",
                "objective": "Minimise total cable cost of the chosen links (Minimum Spanning Tree).",
                "size": {"V": n, "E": e},
                "facts": {"V": n, "E": e, "density": e / full if full else 0.0}}

    if goal == "all_pairs":
        raw = (p.get("matrix") or "").strip()
        if raw:
            m = parse_matrix(raw)
            src = "user-entered"
        else:
            m = random_cost_matrix(_int(p, "zones", 12, 2, 150), 0.5, _int(p, "seed", 3, 0, 10**6))
            src = "randomly generated"
        n = len(m)
        links = sum(1 for i in range(n) for j in range(n) if i != j and m[i][j] != INF)
        neg = any(m[i][j] < 0 for i in range(n) for j in range(n))
        return {"goal": goal, "data": {"matrix": m},
                "entities": f"{n} city zones; {links} directed travel-cost links — {src}",
                "constraints": "Costs may be asymmetric; INF = no direct link.",
                "objective": "Cheapest travel cost for every ordered pair of zones.",
                "size": {"V": n}, "facts": {"V": n, "links": links, "negative_edges": neg,
                                              "all_pairs_required": True}}

    if goal == "budget_projects":
        items = parse_items(p.get("items") or "")
        cap = _float(p, "capacity", 50, 0, 10**9)
        divisible = str(p.get("divisible", "no")).lower() in ("yes", "true", "1")
        integral = abs(cap - round(cap)) < 1e-9 and all(abs(w - round(w)) < 1e-9 for w, _, _ in items)
        return {"goal": goal, "data": {"items": items, "capacity": cap},
                "entities": f"{len(items)} candidate projects (cost, benefit); budget {cap:g}",
                "constraints": "Total cost ≤ budget. Projects are " +
                              ("DIVISIBLE (can be part-funded)." if divisible else "INDIVISIBLE (fund fully or not at all)."),
                "objective": "Maximise total benefit.",
                "size": {"n": len(items), "W": int(cap)},
                "facts": {"n": len(items), "W": cap, "divisible": divisible, "integral": integral}}

    if goal == "city_records":
        n = _int(p, "n", 5000, 10, 200000)
        ratio = _float(p, "searches_per_insert", 5, 0, 50)
        t = _int(p, "btree_t", 16, 2, 256)
        storage = p.get("storage", "disk")
        if storage not in ("disk", "memory"):
            raise ValueError("storage must be 'disk' or 'memory'.")
        return {"goal": goal, "data": {"n": n, "ratio": ratio, "t": t},
                "entities": f"{n} sensor records with ordered integer keys; {int(n * ratio)} lookups",
                "constraints": f"Records live {'on disk/paged storage' if storage == 'disk' else 'in RAM'}; keys are unique and ordered.",
                "objective": "Fast search and insert with few node/page accesses.",
                "size": {"n": n}, "facts": {"n": n, "storage": storage, "search_ratio": ratio, "t": t}}

    if goal == "maintenance_priorities":
        n = _int(p, "n", 3000, 10, 100000)
        ratio = _float(p, "decrease_ratio", 2, 0, 20)
        return {"goal": goal, "data": {"n": n, "ratio": ratio},
                "entities": f"{n} maintenance events with priorities; {int(n * ratio)} priority escalations (decrease-key); all events finally served in order",
                "constraints": "Priorities only ever get more urgent; heaps must support merging.",
                "objective": "Minimise total heap work for insert + escalate + serve.",
                "size": {"n": n}, "facts": {"n": n, "decrease_ratio": ratio}}

    if goal == "sensor_placement":
        n = _int(p, "n", 8, 1, 14)
        return {"goal": goal, "data": {"n": n},
                "entities": f"{n} sensors on an {n}×{n} grid of candidate sites",
                "constraints": "No two sensors may share a row, column or diagonal (they would interfere).",
                "objective": "Find a placement of all n sensors with no conflicts (N-Queen model).",
                "size": {"n": n}, "facts": {"n": n}}

    if goal == "inspection_tour":
        raw = (p.get("matrix") or "").strip()
        if raw:
            m = parse_matrix(raw)
            src = "user-entered"
        else:
            m = random_tour_matrix(_int(p, "sites", 8, 2, 20), _int(p, "seed", 5, 0, 10**6))
            src = "randomly generated"
        n = len(m)
        return {"goal": goal, "data": {"matrix": m},
                "entities": f"{n} sites to inspect, pairwise travel costs — {src}",
                "constraints": "Visit every site exactly once and return to site 0.",
                "objective": "Minimise total tour cost (TSP).",
                "size": {"n": n}, "facts": {"n": n}}

    if goal == "route_feasibility":
        n = _int(p, "nodes", 8, 2, 16)
        raw = (p.get("edges") or "").strip()
        if not raw:
            raise ValueError("Enter the road links, e.g. 0-1;1-2;2-3")
        edges = parse_edges(raw, n, weighted=False)
        cyc = str(p.get("cycle", "yes")).lower() in ("yes", "true", "1")
        return {"goal": goal, "data": {"n": n, "edges": edges, "cycle": cyc},
                "entities": f"{n} zones, {len(edges)} direct road links",
                "constraints": "Each zone visited exactly once" + (" and the route returns to its start." if cyc else "."),
                "objective": "Decide feasibility and output a route if one exists (Hamiltonian formulation).",
                "size": {"V": n, "E": len(edges)}, "facts": {"n": n, "E": len(edges), "cycle": cyc}}

    if goal == "matrix_chain":
        try:
            dims = [int(x) for x in (p.get("dims") or "").replace(";", ",").split(",") if x.strip()]
        except ValueError:
            raise ValueError("Dimensions must be whole numbers, e.g. 10,30,5,60")
        if len(dims) < 3 or len(dims) > 40:
            raise ValueError("Give between 3 and 40 dimensions (2–39 matrices).")
        return {"goal": goal, "data": {"dims": dims},
                "entities": f"{len(dims) - 1} matrices in a processing pipeline",
                "constraints": "Matrix multiplication is associative; order is free but the chain order is fixed.",
                "objective": "Minimise scalar multiplications (proxy for computational load).",
                "size": {"n": len(dims) - 1}, "facts": {"n": len(dims) - 1}}

    raise ValueError(f"Unknown goal '{goal}'.")
