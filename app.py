"""Algorithm 2035 — Flask front end. All logic lives in framework/ and algorithms/."""
import math

from flask import Flask, render_template, request

from framework.pipeline import run, kv, table, text, bars
from framework.problem_model import GOALS
from framework import complexity as C
from algorithms.search import tsp_backtracking, tsp_branch_and_bound
from framework.problem_model import random_tour_matrix
from framework.metrics import timed
import experiments.benchmark as B

app = Flask(__name__)

# Form definitions: (name, label, kind, default, help)
MODULES = {
    "connect_sites": {
        "module": "1 · Infrastructure connection", "algos": "Prim · Kruskal",
        "fields": [("nodes", "Number of sites", "number", "30", "2–400"),
                   ("edges", "Candidate links (optional)", "textarea", "",
                    "Format: from,to,cost;from,to,cost  — leave empty to generate a random graph"),
                   ("density", "Density of generated graph (0–1)", "number", "0.1", "0.05 = sparse, 0.9 = dense")]},
    "all_pairs": {
        "module": "2 · City-wide connectivity", "algos": "Floyd–Warshall",
        "fields": [("zones", "Number of zones (if no matrix)", "number", "12", "2–150"),
                   ("matrix", "Cost matrix (optional)", "textarea", "",
                    "Rows separated by ';', values by ','. INF = no link. Example: 0,3,INF;8,0,2;5,INF,0")]},
    "budget_projects": {
        "module": "3 · Budgeted project selection", "algos": "Fractional · 0/1 Knapsack",
        "fields": [("items", "Projects: cost,benefit,name", "textarea",
                    "10,60,Solar streetlights;20,100,Road repair;30,120,Sensor network",
                    "Separate projects with ';'"),
                   ("capacity", "Budget", "number", "50", "Whole number for the 0/1 DP"),
                   ("divisible", "Can projects be part-funded?", "select:no,yes", "no",
                    "yes → divisible resource, no → all-or-nothing projects")]},
    "city_records": {
        "module": "4a · City records index", "algos": "B-Tree · Red-Black Tree",
        "fields": [("n", "Number of sensor records", "number", "5000", "10–200000"),
                   ("storage", "Where are records stored?", "select:disk,memory", "disk",
                    "disk/paged → minimise page reads; memory → RAM-only"),
                   ("searches_per_insert", "Lookups per insert", "number", "5", "read/write mix"),
                   ("btree_t", "B-Tree minimum degree t", "number", "16", "node holds t-1 … 2t-1 keys")]},
    "maintenance_priorities": {
        "module": "4b · Maintenance priorities", "algos": "Binomial · Fibonacci Heap",
        "fields": [("n", "Number of maintenance events", "number", "3000", "10–100000"),
                   ("decrease_ratio", "Priority escalations per event", "number", "2",
                    "decrease-key operations ÷ events")]},
    "sensor_placement": {
        "module": "5a · Sensor placement", "algos": "N-Queen · Backtracking",
        "fields": [("n", "Grid size n (n sensors)", "number", "8", "1–14")]},
    "inspection_tour": {
        "module": "5b · Inspection tour", "algos": "TSP · Backtracking · Branch & Bound",
        "fields": [("sites", "Number of sites (if no matrix)", "number", "9", "2–20; exact search runs up to 13"),
                   ("matrix", "Travel-cost matrix (optional)", "textarea", "", "Square matrix, INF = no road")]},
    "route_feasibility": {
        "module": "5c · Route feasibility", "algos": "Hamiltonian · Backtracking",
        "fields": [("nodes", "Number of zones", "number", "6", "2–16"),
                   ("edges", "Road links", "textarea", "0-1;1-2;2-3;3-4;4-5;5-0;0-3", "Format: 0-1;1-2;…"),
                   ("cycle", "Must return to the start?", "select:yes,no", "yes", "cycle vs path")]},
    "matrix_chain": {
        "module": "Bonus · Processing pipeline", "algos": "Matrix Chain DP",
        "fields": [("dims", "Matrix dimensions (n matrices → n+1 numbers)", "text", "40,20,30,10,30",
                    "Example 40,20,30,10,30 means A1=40×20, A2=20×30 …")]},
}


def _defaults(goal):
    return {f[0]: f[3] for f in MODULES[goal]["fields"]}


def _nav(active):
    return active


@app.route("/", methods=["GET"])
def home():
    cards = [(g, MODULES[g]["module"], GOALS[g], MODULES[g]["algos"]) for g in MODULES]
    return render_template("home.html", cards=cards, active="home")


@app.route("/solve/<goal>", methods=["GET", "POST"])
def solve(goal):
    if goal not in MODULES:
        return render_template("page.html", active="home", title="Not found",
                               error="Unknown module."), 404
    values = _defaults(goal)
    result = error = None
    if request.method == "POST":
        values = {f[0]: request.form.get(f[0], "").strip() for f in MODULES[goal]["fields"]}
        try:
            result = run(goal, values)
        except Exception as exc:                  # show input problems to the user
            error = str(exc)
    return render_template("page.html", active=goal, goal=goal, mod=MODULES[goal], title=GOALS[goal],
                           values=values, result=result, error=error)


ADAPT_CASES = [
    ("Change the graph density → different MST algorithm", "connect_sites",
     ("Sparse: 100 sites, density 4%", {"nodes": "100", "density": "0.04"}),
     ("Dense: 100 sites, density 90%", {"nodes": "100", "density": "0.9"})),
    ("Change divisibility → different knapsack technique", "budget_projects",
     ("Indivisible projects", {"items": "10,60,Solar;20,100,Roads;30,120,Sensors", "capacity": "50", "divisible": "no"}),
     ("Divisible budget", {"items": "10,60,Solar;20,100,Roads;30,120,Sensors", "capacity": "50", "divisible": "yes"})),
    ("Change the storage medium → different index structure", "city_records",
     ("Records on disk", {"n": "3000", "storage": "disk"}),
     ("Records in memory", {"n": "3000", "storage": "memory"})),
    ("Change the update pattern → different heap", "maintenance_priorities",
     ("Few escalations, small system", {"n": "500", "decrease_ratio": "0.2"}),
     ("Many escalations, large system", {"n": "3000", "decrease_ratio": "3"})),
    ("Change the number of sites → different tour method (and a scalability limit)", "inspection_tour",
     ("6 sites", {"sites": "6"}),
     ("10 sites", {"sites": "10"})),
    ("Too many sites → exact search refused", "inspection_tour",
     ("10 sites", {"sites": "10"}),
     ("16 sites", {"sites": "16"})),
]


@app.route("/adapt")
def adapt():
    blocks = []
    for title, goal, a, b in ADAPT_CASES:
        rows = []
        for label, params in (a, b):
            r = run(goal, params)
            rows.append([label, r["decision"]["problem_type"], "★ " + r["decision"]["chosen"], r["decision"]["rule"].split(" Prim (binary heap)")[0]])
        blocks.append(table(title, ["Scenario", "Recognised problem", "Chosen technique", "Rule that fired"], rows))
    intro = text("What this shows",
                 "The same goal is submitted twice with one property changed. The decision engine looks only at problem properties (density, divisibility, storage, escalation ratio, number of sites) — never an algorithm name — and selects a different technique each time.")
    return render_template("page.html", active="adapt", title="Adaptation demo", blocks=[intro] + blocks)


@app.route("/experiments")
def experiments():
    blocks = [text("About these experiments",
                   "Each table is generated live from the real implementations. Sizes here are kept small so the page loads quickly; "
                   "run `python experiments/benchmark.py` for larger sizes and CSV files in experiments/results/. "
                   "Wall-clock times depend on your machine; operation counts are deterministic.")]
    specs = [
        ("MST: sparse → dense (V = 50, 100, 200)", B.mst_experiment, "Operation counts follow theory: Prim-matrix is flat in E (2V²), Kruskal grows with E·log E. The rule switches when they cross. Wall-clock in Python favours Kruskal because sorting is native C — a caveat, not a contradiction."),
        ("Floyd–Warshall: growth with zones", B.floyd_experiment, "ops/V³ = 1.0 exactly; doubling V multiplies time by ≈ 8."),
        ("TSP: Backtracking vs Branch & Bound", B.tsp_experiment, "Backtracking generates ~(n-1)! states; Branch & Bound prunes most. Both still grow explosively — NP-Hard."),
        ("Red-Black Tree vs B-Tree", lambda: B.tree_experiment((1000, 5000)), "B-Tree is 3–4 levels deep versus 12–15, so far fewer node (page) visits — but more in-node comparisons, so in RAM it is not faster."),
        ("Binomial vs Fibonacci Heap", lambda: B.heap_experiment((1000, 3000), (0.0, 1.0, 4.0)), "Fibonacci inserts in O(1); its decrease-key advantage grows with escalation ratio."),
        ("Knapsack: DP cost grows with budget W", B.knapsack_experiment, "0/1 DP cells = n·W grow with the NUMBER W (pseudo-polynomial); fractional greedy stays tiny. Fractional value ≥ 0/1 value always."),
        ("N-Queen backtracking", B.nqueens_experiment, "States for the first solution stay tiny compared with n! — early conflict pruning."),
    ]
    for title, fn, note in specs:
        rows = fn()
        headers = list(rows[0].keys())
        blocks.append(table(title, headers, [[r[h] for h in headers] for r in rows], note=note))
        if title.startswith("Floyd"):
            blocks.append(bars("Floyd–Warshall operations", [str(r["V"]) for r in rows], [r["ops"] for r in rows]))
        if title.startswith("TSP"):
            blocks.append(bars("TSP states generated by Branch & Bound", [str(r["n"]) for r in rows], [r["bb_states"] for r in rows]))
    return render_template("page.html", active="experiments", title="Experiments", blocks=blocks)


@app.route("/complexity")
def complexity():
    blocks = []
    blocks.append(table("Time and space complexity of every algorithm and data structure used",
                        ["Technique", "Time", "Space", "Correctness idea", "Class"],
                        [[k, v["time"], v["space"], v["correctness"], v["class"]] for k, v in C.ALGORITHMS.items()]))
    rows = []
    for n in range(4, 11):
        m = random_tour_matrix(n, seed=n)
        bt, t1 = timed(tsp_backtracking, m, repeat=1)
        bb, t2 = timed(tsp_branch_and_bound, m, repeat=1)
        rows.append([n, f"{math.factorial(n - 1):,}", f"{bt['states_generated']:,}", f"{bb['states_generated']:,}", round(t1, 2), round(t2, 2)])
    blocks.append(table("Where NP-Hard behaviour appears: exact TSP as sites are added",
                        ["Sites n", "(n-1)! tours", "Backtracking states", "Branch & Bound states", "BT ms", "B&B ms"], rows,
                        note="Each extra site multiplies the tour count by n. At n = 20 there would be 1.2 × 10¹⁷ tours — hence the framework refuses exact TSP beyond 13 sites."))
    blocks.append(bars("Backtracking states (measured)", [str(r[0]) for r in rows], [int(r[2].replace(",", "")) for r in rows]))
    blocks.append(table("Pseudo-polynomial: 0/1 Knapsack DP table size n·W (n = 50)",
                        ["Budget W", "DP cells"], [[w, f"{51 * (w + 1):,}"] for w in (100, 1000, 10000, 100000, 10**6)],
                        note="Polynomial in the NUMBER W but exponential in its bit-length (~log₂W bits) — pseudo-polynomial."))
    blocks.append(table("NP-Hard / NP-Complete classification", ["Problem", "Statement"],
                        [[k, v] for k, v in C.NP_STATEMENTS.items()]))
    blocks.append(table("Gaps in classic algorithms that motivated newer ones", ["Classic technique", "Gap", "What was developed"],
                        [list(g) for g in C.GAPS]))
    return render_template("page.html", active="complexity", title="Complexity awareness", blocks=blocks)


if __name__ == "__main__":
    app.run(debug=True)
