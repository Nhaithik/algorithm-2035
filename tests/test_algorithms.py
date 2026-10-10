"""Run:  python -m unittest discover -s tests -v
Covers normal, boundary, large and difficult cases for every algorithm,
data structure, decision rule, pipeline module and web route."""
import itertools
import math
import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algorithms.graphs import prim_heap, prim_matrix, kruskal_mst, floyd_warshall, INF
from algorithms.knapsack import fractional_knapsack, zero_one_knapsack, greedy_zero_one
from algorithms.search import (solve_n_queens, hamiltonian, tsp_backtracking,
                               tsp_branch_and_bound, matrix_chain)
from algorithms.structures import (RedBlackTree, BTree, BinomialHeap, FibonacciHeap,
                                   tree_workload, heap_workload)
from framework.decision_engine import decide
from framework.pipeline import run
from framework.problem_model import (random_graph, random_tour_matrix, parse_matrix,
                                     parse_edges, build_model, GOALS)


# ------------------------------------------------------------------ MST
class MSTTests(unittest.TestCase):
    def test_all_three_agree_on_random_graphs(self):
        for seed in range(15):
            n = random.Random(seed).randint(2, 40)
            edges = random_graph(n, random.Random(seed).random(), seed)
            costs = {prim_matrix(n, edges)["total_cost"], prim_heap(n, edges)["total_cost"],
                     kruskal_mst(n, edges)["total_cost"]}
            self.assertEqual(len(costs), 1)

    def test_known_answer(self):
        edges = [(0, 1, 4), (0, 2, 3), (1, 2, 1), (1, 3, 2), (2, 3, 4), (3, 4, 2), (2, 4, 5)]
        for f in (prim_matrix, prim_heap, kruskal_mst):
            r = f(5, edges)
            self.assertEqual(r["total_cost"], 8)
            self.assertTrue(r["connected"])
            self.assertEqual(len(r["edges"]), 4)

    def test_boundary_single_node_and_disconnected(self):
        for f in (prim_matrix, prim_heap, kruskal_mst):
            self.assertEqual(f(1, [])["total_cost"], 0)
            r = f(4, [(0, 1, 2), (2, 3, 5)])
            self.assertFalse(r["connected"])
            self.assertEqual(r["components"], 2)
            self.assertEqual(r["total_cost"], 7)

    def test_parallel_edges_use_cheapest(self):
        for f in (prim_matrix, prim_heap, kruskal_mst):
            self.assertEqual(f(2, [(0, 1, 9), (0, 1, 3)])["total_cost"], 3)

    def test_invalid_input(self):
        with self.assertRaises(ValueError):
            kruskal_mst(0, [])
        with self.assertRaises(ValueError):
            prim_heap(3, [(0, 7, 1)])

    def test_counting_modes_do_not_change_result(self):
        edges = random_graph(30, 0.3)
        self.assertEqual(kruskal_mst(30, edges)["total_cost"], kruskal_mst(30, edges, True)["total_cost"])
        self.assertEqual(prim_heap(30, edges)["total_cost"], prim_heap(30, edges, True)["total_cost"])
        self.assertGreater(kruskal_mst(30, edges, True)["ops"]["sort_comparisons"], 0)

    def test_large_complete_graph(self):
        edges = random_graph(150, 1.0)
        self.assertEqual(prim_matrix(150, edges)["total_cost"], kruskal_mst(150, edges)["total_cost"])


# ---------------------------------------------------------- Floyd-Warshall
class FloydTests(unittest.TestCase):
    def test_known(self):
        d, info = floyd_warshall([[0, 3, INF], [INF, 0, 2], [4, INF, 0]])
        self.assertEqual(d[0][2], 5)
        self.assertEqual(d[2][1], 7)
        self.assertEqual(info["ops"], 27)

    def test_ops_are_v_cubed(self):
        for n in (1, 5, 17):
            m = [[0 if i == j else 1 for j in range(n)] for i in range(n)]
            self.assertEqual(floyd_warshall(m)[1]["ops"], n ** 3)

    def test_matches_bruteforce_relaxation(self):
        rnd = random.Random(2)
        n = 8
        m = [[0 if i == j else rnd.choice([INF, rnd.randint(1, 20)]) for j in range(n)] for i in range(n)]
        d, _ = floyd_warshall(m)
        ref = [r[:] for r in m]
        for _ in range(n):                       # Bellman-Ford style repeated relaxation
            for i in range(n):
                for j in range(n):
                    for k in range(n):
                        ref[i][j] = min(ref[i][j], ref[i][k] + m[k][j])
        self.assertEqual(d, ref)

    def test_negative_cycle_flag_and_unreachable(self):
        _, info = floyd_warshall([[0, 1], [-3, 0]])
        self.assertTrue(info["negative_cycle"])
        d, _ = floyd_warshall([[0, INF], [INF, 0]])
        self.assertEqual(d[0][1], INF)

    def test_rejects_non_square(self):
        with self.assertRaises(ValueError):
            floyd_warshall([[0, 1], [1]])


# ----------------------------------------------------------------- Knapsack
class KnapsackTests(unittest.TestCase):
    ITEMS = [(10, 60, "a"), (20, 100, "b"), (30, 120, "c")]

    def test_classic(self):
        self.assertEqual(zero_one_knapsack(self.ITEMS, 50)["total_value"], 220)
        self.assertEqual(fractional_knapsack(self.ITEMS, 50)["total_value"], 240)
        self.assertEqual(greedy_zero_one(self.ITEMS, 50)["total_value"], 160)   # greedy is NOT optimal

    def test_fractional_is_upper_bound_on_01(self):
        rnd = random.Random(1)
        for _ in range(30):
            items = [(rnd.randint(1, 15), rnd.randint(1, 50), f"i{i}") for i in range(rnd.randint(1, 8))]
            W = rnd.randint(0, 40)
            self.assertGreaterEqual(fractional_knapsack(items, W)["total_value"] + 1e-9,
                                    zero_one_knapsack(items, W)["total_value"])

    def test_01_matches_bruteforce(self):
        rnd = random.Random(5)
        for _ in range(30):
            items = [(rnd.randint(1, 10), rnd.randint(0, 30), f"i{i}") for i in range(rnd.randint(1, 9))]
            W = rnd.randint(0, 30)
            best = max(sum(items[i][1] for i in c) for r in range(len(items) + 1)
                       for c in itertools.combinations(range(len(items)), r)
                       if sum(items[i][0] for i in c) <= W)
            r = zero_one_knapsack(items, W)
            self.assertEqual(r["total_value"], best)
            self.assertLessEqual(r["capacity_used"], W)

    def test_boundaries(self):
        self.assertEqual(zero_one_knapsack(self.ITEMS, 0)["total_value"], 0)
        self.assertEqual(fractional_knapsack(self.ITEMS, 0)["total_value"], 0)
        self.assertEqual(zero_one_knapsack([], 10)["total_value"], 0)
        self.assertEqual(zero_one_knapsack([(100, 5, "x")], 10)["selected"], [])
        self.assertEqual(fractional_knapsack([(100, 50, "x")], 10)["total_value"], 5)

    def test_validation(self):
        with self.assertRaises(ValueError):
            zero_one_knapsack([(1.5, 3, "x")], 4)
        with self.assertRaises(ValueError):
            zero_one_knapsack([(1, 3, "x")], 4.5)
        with self.assertRaises(ValueError):
            fractional_knapsack([(0, 3, "x")], 4)
        with self.assertRaises(ValueError):
            zero_one_knapsack([(1, 1, "x")], 10 ** 6)

    def test_dp_cells_grow_with_W(self):
        a = zero_one_knapsack(self.ITEMS, 100)["ops"]
        b = zero_one_knapsack(self.ITEMS, 1000)["ops"]
        self.assertGreater(b, 9 * a)


# ------------------------------------------------------------ Data structures
class TreeTests(unittest.TestCase):
    def test_rbt_and_btree_invariants(self):
        for n in (0, 1, 2, 10, 500, 4000):
            keys = random.Random(n).sample(range(10 * n + 5), n)
            rb, b3, b16 = RedBlackTree(), BTree(2), BTree(16)
            for k in keys:
                rb.insert(k); b3.insert(k); b16.insert(k)
            for t in (rb, b3, b16):
                self.assertTrue(t.is_valid())
                self.assertEqual(t.inorder(), sorted(keys))
            for k in keys[:100]:
                self.assertEqual(rb.search(k), k)
                self.assertEqual(b3.search(k), k)
            self.assertIsNone(rb.search(-5))
            self.assertIsNone(b16.search(-5))

    def test_sorted_input_stays_balanced(self):           # difficult case: worst case for a plain BST
        rb, bt = RedBlackTree(), BTree(8)
        for k in range(5000):
            rb.insert(k); bt.insert(k)
        self.assertLessEqual(rb.height(), 2 * math.log2(5001))
        self.assertLessEqual(bt.height(), 5)
        self.assertTrue(rb.is_valid() and bt.is_valid())

    def test_duplicate_insert_updates(self):
        for t in (RedBlackTree(), BTree(3)):
            self.assertTrue(t.insert(1, "a"))
            self.assertFalse(t.insert(1, "b"))
            self.assertEqual(t.search(1), "b")
            self.assertEqual(t.size, 1)

    def test_btree_is_shallower_than_rbt(self):
        rb, bt = RedBlackTree(), BTree(16)
        for k in random.Random(1).sample(range(100000), 20000):
            rb.insert(k); bt.insert(k)
        self.assertLess(bt.height(), rb.height())

    def test_btree_bad_degree(self):
        with self.assertRaises(ValueError):
            BTree(1)

    def test_workload_rows(self):
        rb, bt = tree_workload(500, 2, 8)
        self.assertTrue(rb["valid"] and bt["valid"])
        self.assertEqual(rb["hits"], rb["queries"])
        self.assertLess(bt["search_node_visits"], rb["search_node_visits"])


class HeapTests(unittest.TestCase):
    def _fuzz(self, cls):
        rnd = random.Random(99)
        for _ in range(40):
            h, cur, nxt = cls(), {}, 0
            for _ in range(rnd.randint(0, 200)):
                op = rnd.random()
                if op < 0.5 or not cur:
                    k = rnd.randint(0, 500); h.insert(k, nxt); cur[nxt] = k; nxt += 1
                elif op < 0.75:
                    i = rnd.choice(list(cur)); nk = cur[i] - rnd.randint(0, 40)
                    h.decrease_key(i, nk); cur[i] = nk
                else:
                    k, i = h.extract_min()
                    self.assertEqual(k, min(cur.values()))
                    self.assertEqual(cur.pop(i), k)
                self.assertEqual(len(h), len(cur))
            rest = []
            while len(h):
                k, i = h.extract_min(); self.assertEqual(cur.pop(i), k); rest.append(k)
            self.assertEqual(rest, sorted(rest))
            self.assertFalse(cur)

    def test_binomial_fuzz(self):
        self._fuzz(BinomialHeap)

    def test_fibonacci_fuzz(self):
        self._fuzz(FibonacciHeap)

    def test_merge(self):
        for cls in (BinomialHeap, FibonacciHeap):
            a, b = cls(), cls()
            for i in range(60):
                a.insert(random.randint(0, 99), i)
            for i in range(60, 130):
                b.insert(random.randint(0, 99), i)
            a.merge(b)
            self.assertEqual(len(a), 130)
            self.assertEqual(len(b), 0)
            out = [a.extract_min()[0] for _ in range(130)]
            self.assertEqual(out, sorted(out))

    def test_empty_and_bad_decrease(self):
        for cls in (BinomialHeap, FibonacciHeap):
            h = cls()
            with self.assertRaises(IndexError):
                h.extract_min()
            h.insert(5, "x")
            with self.assertRaises(ValueError):
                h.decrease_key("x", 9)
            h.merge(cls())                         # merging empty is harmless
            self.assertEqual(h.find_min(), (5, "x"))

    def test_fibonacci_insert_is_cheaper(self):
        b, f = BinomialHeap(), FibonacciHeap()
        for i in range(2000):
            b.insert(i, i); f.insert(i, i)
        self.assertLess(f.ops, b.ops)

    def test_workload_sorted(self):
        for r in heap_workload(400, 2.0):
            self.assertTrue(r["sorted_ok"])


# ------------------------------------------------------------ Backtracking / B&B
class SearchTests(unittest.TestCase):
    def test_nqueens_counts(self):
        for n, cnt in ((1, 1), (2, 0), (3, 0), (4, 2), (5, 10), (6, 4), (8, 92)):
            self.assertEqual(solve_n_queens(n, True)["solutions"], cnt)

    def test_nqueens_solution_valid(self):
        for n in (4, 8, 12, 14):
            cols = solve_n_queens(n)["columns"]
            self.assertEqual(len(set(cols)), n)
            self.assertEqual(len({r - c for r, c in enumerate(cols)}), n)
            self.assertEqual(len({r + c for r, c in enumerate(cols)}), n)

    def test_nqueens_limits(self):
        for bad in (0, 15):
            with self.assertRaises(ValueError):
                solve_n_queens(bad)

    def test_hamiltonian(self):
        self.assertTrue(hamiltonian(5, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)])["found"])
        self.assertFalse(hamiltonian(4, [(0, 1), (0, 2), (0, 3)])["found"])        # star: no cycle
        self.assertTrue(hamiltonian(4, [(0, 1), (1, 2), (2, 3)], cycle=False)["found"])
        self.assertFalse(hamiltonian(4, [(0, 1), (2, 3)], cycle=False)["found"])    # disconnected
        r = hamiltonian(6, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (0, 3)])
        self.assertEqual(r["route"][0], r["route"][-1])
        self.assertEqual(sorted(r["route"][:-1]), list(range(6)))

    def test_hamiltonian_path_from_non_zero_start(self):
        self.assertTrue(hamiltonian(3, [(0, 1), (0, 2)], cycle=False)["found"])     # path 1-0-2

    def test_hamiltonian_abort_cap(self):
        n = 12   # complete graph minus an isolated-ish vertex => hard exhaustive search
        edges = [(i, j) for i in range(n - 1) for j in range(i + 1, n - 1)] + [(0, n - 1)]
        r = hamiltonian(n, edges, cycle=True, max_states=5000)
        self.assertTrue(r["aborted"] or not r["found"])

    def test_tsp_known(self):
        m = [[0, 10, 15, 20], [10, 0, 35, 25], [15, 35, 0, 30], [20, 25, 30, 0]]
        for f in (tsp_backtracking, tsp_branch_and_bound):
            r = f(m)
            self.assertEqual(r["cost"], 80)
            self.assertEqual(r["route"][0], 0)
            self.assertEqual(r["route"][-1], 0)

    def test_tsp_matches_bruteforce_incl_missing_roads(self):
        rnd = random.Random(8)
        for _ in range(50):
            n = rnd.randint(2, 7)
            m = [[0 if i == j else rnd.randint(1, 40) for j in range(n)] for i in range(n)]
            for _ in range(rnd.randint(0, 4)):
                i, j = rnd.randrange(n), rnd.randrange(n)
                if i != j:
                    m[i][j] = INF
            best = INF
            for p in itertools.permutations(range(1, n)):
                r = (0,) + p + (0,)
                best = min(best, sum(m[r[i]][r[i + 1]] for i in range(n)))
            exp = None if best == INF else best
            self.assertEqual(tsp_backtracking(m)["cost"], exp)
            self.assertEqual(tsp_branch_and_bound(m)["cost"], exp)

    def test_branch_and_bound_prunes(self):
        m = random_tour_matrix(10, 3)
        bt, bb = tsp_backtracking(m), tsp_branch_and_bound(m)
        self.assertEqual(bt["cost"], bb["cost"])
        self.assertLess(bb["states_generated"], bt["states_generated"] / 10)
        self.assertGreater(bb["states_pruned"], 0)

    def test_state_growth_is_factorial_like(self):
        s = [tsp_backtracking(random_tour_matrix(n, n))["states_generated"] for n in (6, 7, 8, 9)]
        self.assertTrue(all(b > 4 * a for a, b in zip(s, s[1:])))

    def test_tsp_cap_and_validation(self):
        r = tsp_backtracking(random_tour_matrix(9, 1), max_states=100)
        self.assertTrue(r["aborted"])
        with self.assertRaises(ValueError):
            tsp_branch_and_bound([[0]])

    def test_matrix_chain(self):
        r = matrix_chain([10, 30, 5, 60])
        self.assertEqual(r["min_multiplications"], 4500)
        self.assertEqual(matrix_chain([40, 20, 30, 10, 30])["min_multiplications"], 26000)
        self.assertLess(matrix_chain([40, 20, 30, 10, 30])["min_multiplications"],
                        matrix_chain([40, 20, 30, 10, 30])["left_to_right"])
        self.assertEqual(matrix_chain([3, 4])["min_multiplications"], 0)
        with self.assertRaises(ValueError):
            matrix_chain([5])


# --------------------------------------------------------- Decision engine
class DecisionTests(unittest.TestCase):
    def mst(self, n, density, seed=7):
        edges = random_graph(n, density, seed)
        return decide("connect_sites", {"V": n, "E": len(edges), "density": len(edges) / (n * (n - 1) / 2)})

    def test_mst_changes_with_density(self):
        self.assertEqual(self.mst(100, 0.04)["chosen"], "Kruskal")
        self.assertEqual(self.mst(100, 0.9)["chosen"], "Prim (adjacency matrix)")

    def test_knapsack_changes_with_divisibility(self):
        f = {"n": 3, "W": 50, "integral": True}
        self.assertIn("Fractional", decide("budget_projects", dict(f, divisible=True))["chosen"])
        self.assertIn("0/1", decide("budget_projects", dict(f, divisible=False))["chosen"])

    def test_index_changes_with_storage(self):
        f = {"n": 1000, "search_ratio": 5, "t": 16}
        self.assertEqual(decide("city_records", dict(f, storage="disk"))["chosen"], "B-Tree")
        self.assertEqual(decide("city_records", dict(f, storage="memory"))["chosen"], "Red-Black Tree")

    def test_heap_changes_with_workload(self):
        self.assertEqual(decide("maintenance_priorities", {"n": 5000, "decrease_ratio": 3})["chosen"], "Fibonacci Heap")
        self.assertEqual(decide("maintenance_priorities", {"n": 5000, "decrease_ratio": 0.1})["chosen"], "Binomial Heap")
        self.assertEqual(decide("maintenance_priorities", {"n": 50, "decrease_ratio": 3})["chosen"], "Binomial Heap")

    def test_tsp_method_changes_with_size_and_refuses(self):
        self.assertEqual(decide("inspection_tour", {"n": 6})["chosen"], "TSP Backtracking")
        self.assertEqual(decide("inspection_tour", {"n": 11})["chosen"], "TSP Branch & Bound")
        self.assertTrue(decide("inspection_tour", {"n": 18})["chosen"].startswith("REFUSED"))

    def test_every_decision_is_explained(self):
        samples = {
            "connect_sites": {"V": 10, "E": 20, "density": .4}, "all_pairs": {"V": 5, "negative_edges": False},
            "budget_projects": {"n": 3, "W": 10, "divisible": False, "integral": True},
            "city_records": {"n": 100, "storage": "disk", "search_ratio": 1, "t": 4},
            "maintenance_priorities": {"n": 100, "decrease_ratio": 1}, "sensor_placement": {"n": 6},
            "inspection_tour": {"n": 7}, "route_feasibility": {"n": 5, "E": 6, "cycle": True},
            "matrix_chain": {"n": 4}}
        self.assertEqual(set(samples), set(GOALS))
        for g, f in samples.items():
            d = decide(g, f)
            for key in ("problem_type", "chosen", "rejected", "rule", "complexity", "np_note"):
                self.assertTrue(d[key], f"{g} missing {key}")
            self.assertGreaterEqual(len(d["rejected"]), 1)


# ------------------------------------------------------------- Pipeline
class PipelineTests(unittest.TestCase):
    DEFAULTS = {
        "connect_sites": {"nodes": "30", "density": "0.1"}, "all_pairs": {"zones": "8"},
        "budget_projects": {"items": "10,60,a;20,100,b;30,120,c", "capacity": "50", "divisible": "no"},
        "city_records": {"n": "300"}, "maintenance_priorities": {"n": "200", "decrease_ratio": "1"},
        "sensor_placement": {"n": "6"}, "inspection_tour": {"sites": "7"},
        "route_feasibility": {"nodes": "5", "edges": "0-1;1-2;2-3;3-4;4-0"}, "matrix_chain": {"dims": "10,20,30,40"}}

    def test_all_goals_run(self):
        for g in GOALS:
            r = run(g, self.DEFAULTS[g])
            self.assertTrue(r["blocks"] and r["summary"] and r["model"]["entities"], g)

    def test_pipeline_adapts_tour_method(self):
        self.assertEqual(run("inspection_tour", {"sites": "6"})["decision"]["chosen"], "TSP Backtracking")
        self.assertEqual(run("inspection_tour", {"sites": "10"})["decision"]["chosen"], "TSP Branch & Bound")
        refused = run("inspection_tour", {"sites": "16"})
        self.assertTrue(refused["decision"]["chosen"].startswith("REFUSED"))

    def test_bad_inputs(self):
        for g, p in (("connect_sites", {"nodes": "1"}), ("all_pairs", {"matrix": "1,2;3"}),
                     ("budget_projects", {"items": ""}), ("sensor_placement", {"n": "99"}),
                     ("route_feasibility", {"nodes": "4", "edges": ""}), ("matrix_chain", {"dims": "1,2"})):
            with self.assertRaises(ValueError, msg=g):
                run(g, p)
        with self.assertRaises(ValueError):
            run("nope", {})

    def test_parsers(self):
        self.assertEqual(parse_matrix("0,INF;1,0")[0][1], INF)
        self.assertEqual(parse_edges("0-1;1-2", 3, weighted=False), [(0, 1), (1, 2)])
        self.assertEqual(parse_edges("0,0,5;0,1,2", 2), [(0, 1, 2.0)])      # self-loop ignored
        with self.assertRaises(ValueError):
            parse_edges("0,9,1", 3)

    def test_unknown_goal_model(self):
        with self.assertRaises(ValueError):
            build_model("zzz", {})


# ------------------------------------------------------------- Web routes
class WebTests(unittest.TestCase):
    def setUp(self):
        from app import app
        self.c = app.test_client()

    def test_pages(self):
        for url in ("/", "/adapt", "/experiments", "/complexity"):
            self.assertEqual(self.c.get(url).status_code, 200, url)

    def test_each_module_get_and_post(self):
        from app import MODULES, _defaults
        for g in MODULES:
            self.assertEqual(self.c.get(f"/solve/{g}").status_code, 200)
            r = self.c.post(f"/solve/{g}", data=_defaults(g))
            self.assertEqual(r.status_code, 200)
            self.assertIn(b"Decision engine", r.data, g)

    def test_error_is_shown_not_crashed(self):
        r = self.c.post("/solve/sensor_placement", data={"n": "500"})
        self.assertEqual(r.status_code, 200)
        self.assertIn(b"Input problem", r.data)

    def test_unknown_module_404(self):
        self.assertEqual(self.c.get("/solve/unknown").status_code, 404)


if __name__ == "__main__":
    unittest.main()
