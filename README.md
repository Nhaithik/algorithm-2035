# Algorithm 2035 — Adaptive Algorithm Selection Framework for Smart Cities

DAA Assignment 1 · Python + Flask · every algorithm and data structure is implemented from scratch
(only `heapq` is used for the fast path of heap-based Prim; a hand-written counting heap is used for operation counts).

A smart-city request goes in. The framework **models** it, **chooses** a DAA technique from the
problem's *properties* (never from an algorithm name typed by the user), **runs** it against the
alternative, and **explains** the decision with complexity and NP-class notes.

```
SCENARIO INPUT → PROBLEM MODEL → DECISION ENGINE → EXECUTE & COMPARE → ANALYSIS → EXPLANATION / ADAPTATION
```

## Modules

| # | Module | Techniques (all implemented) | Decision is driven by |
|---|---|---|---|
| 1 | Infrastructure connection | Prim (matrix), Prim (heap), Kruskal | V, E → modelled cost `2V²` vs `E·log₂E` |
| 2 | City-wide connectivity | Floyd–Warshall (+ scaling table) | all-pairs requested |
| 3 | Budgeted project selection | Fractional Knapsack, 0/1 Knapsack DP (+ ratio-greedy counter-example) | divisible or not |
| 4a | City records index | Red-Black Tree vs B-Tree | disk vs memory |
| 4b | Maintenance priorities | Binomial Heap vs Fibonacci Heap | decrease-key ratio, n |
| 5a | Sensor placement | N-Queen backtracking | pairwise grid conflicts |
| 5b | Inspection tour | TSP Backtracking vs Branch & Bound | number of sites (refuses > 13) |
| 5c | Route feasibility | Hamiltonian cycle/path backtracking | feasibility, no cost |
| 6 | Complexity awareness | measured TSP growth, pseudo-polynomial DP table, NP table, gaps table | — |
| Bonus | Processing pipeline | Matrix Chain Multiplication DP | — |

Extra pages: **Adaptation demo** (same goal, one property changed → different algorithm),
**Experiments** (live tables + bars), **Complexity** (time/space of everything, NP-Hard/NP-Complete statements, gaps that led to newer algorithms).

## Run (macOS / VS Code)

```bash
cd algorithm-2035
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```
Open http://127.0.0.1:5000 (on macOS, if port 5000 is busy because of AirPlay Receiver, run `flask --app app run --port 5001`).

Windows: `python -m venv .venv`, `.venv\Scripts\activate`, then the same `pip` / `python app.py`.

## Tests (57)

```bash
python -m unittest discover -s tests -v
```
Normal, boundary, large and difficult cases; algorithms are cross-checked against brute force
(TSP vs all permutations, 0/1 Knapsack vs all subsets, N-Queen known counts, Floyd–Warshall vs repeated relaxation,
heaps/trees against sorted reference and structural-invariant checkers).

## Experiments (CSV output)

```bash
python experiments/benchmark.py
```
Writes `experiments/results/*.csv` (MST sparse→dense, Floyd–Warshall growth, TSP BT vs B&B, trees, heaps, knapsack, N-Queen).
Operation counts are deterministic; timings depend on your machine.

## Input formats

| Field | Format |
|---|---|
| Graph links | `from,to,cost;from,to,cost` (sites numbered from 0). Leave empty to generate a random graph |
| Cost matrix | `0,3,INF;8,0,2;5,INF,0` — rows split by `;`, `INF` = no link |
| Projects | `cost,benefit,name;cost,benefit,name` |
| Road links (Hamiltonian) | `0-1;1-2;2-3` |
| Matrix dimensions | `40,20,30,10,30` (n matrices → n+1 numbers) |

## Project layout

```
app.py                       Flask routes (no algorithm logic)
framework/
  problem_model.py           stage 2 – parse input, build model (entities, constraints, objective, size)
  decision_engine.py         stage 3 – property-based rules with rejected alternatives + reasons
  pipeline.py                stages 4-5 – execute, compare, explain
  complexity.py              time/space, correctness ideas, NP statements, gaps
  metrics.py                 timing helper
algorithms/
  graphs.py  knapsack.py  structures.py  search.py
experiments/benchmark.py     experiments + CSV writer
tests/test_algorithms.py     57 tests
docs/architecture.md         architecture diagram
docs/DAA_analysis.md         analysis, correctness, NP discussion, gaps, limitations, viva Q&A
```

## Honest notes (read before the viva)

* **No algorithm is claimed to be always faster.** The MST rule uses a cost model; the page also shows
  measured operations *and* wall-clock. In CPython, Kruskal often wins wall-clock even where the model predicts
  Prim-matrix, because `sorted()` runs in native C while the matrix scan is interpreted Python.
  Operation counts, which follow the theory, are the primary evidence.
* **Thresholds are design choices**, not laws: heap rule (`n ≥ 1000` and ≥ 1 decrease-key per event),
  TSP limits (≤ 8 backtracking, 9–13 branch & bound, > 13 refused). They are named constants in `framework/decision_engine.py`.
* **0/1 Knapsack** needs integer costs and budget (≤ 20000) — scale your numbers to integers.
* **Complexity labels:** optimisation TSP is NP-Hard (decision TSP NP-Complete); 0/1 Knapsack DP is
  pseudo-polynomial (decision form NP-Complete); standard N-Queen is *not* labelled NP-Complete.
* The Fibonacci Heap is implemented with circular linked lists for true O(1) cuts, but its constants are large — which is why the engine
  does not always pick it.

## Upload to GitHub

```bash
git init
git add .
git commit -m "Algorithm 2035: adaptive DAA framework for smart cities"
git branch -M main
git remote add origin https://github.com/<your-username>/algorithm-2035.git
git push -u origin main
```
Add screenshots of each module, the test output, and the experiment tables to your report.
