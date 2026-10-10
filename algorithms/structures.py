"""Executable implementations of the four advanced data structures:
Red-Black Tree, B-Tree, Binomial Heap, Fibonacci Heap -- each with counters
so the framework can compare real operation counts, not only textbook bounds."""
import random
import time

# ----------------------------------------------------------------------------
# Red-Black Tree  (CLRS).  search/insert O(log n), height <= 2 log2(n+1),
# at most 2 rotations per insert.  Space O(n).
# ----------------------------------------------------------------------------
RED, BLACK = 0, 1


class _RBNode:
    __slots__ = ("key", "value", "color", "left", "right", "parent")

    def __init__(self, key, value, color):
        self.key, self.value, self.color = key, value, color
        self.left = self.right = self.parent = None


class RedBlackTree:
    def __init__(self):
        self.nil = _RBNode(None, None, BLACK)
        self.nil.left = self.nil.right = self.nil.parent = self.nil
        self.root = self.nil
        self.size = 0
        self.rotations = 0
        self.comparisons = 0
        self.node_visits = 0

    def _rotate_left(self, x):
        y = x.right
        x.right = y.left
        if y.left is not self.nil:
            y.left.parent = x
        y.parent = x.parent
        if x.parent is self.nil:
            self.root = y
        elif x is x.parent.left:
            x.parent.left = y
        else:
            x.parent.right = y
        y.left = x
        x.parent = y
        self.rotations += 1

    def _rotate_right(self, x):
        y = x.left
        x.left = y.right
        if y.right is not self.nil:
            y.right.parent = x
        y.parent = x.parent
        if x.parent is self.nil:
            self.root = y
        elif x is x.parent.right:
            x.parent.right = y
        else:
            x.parent.left = y
        y.right = x
        x.parent = y
        self.rotations += 1

    def search(self, key):
        x = self.root
        while x is not self.nil:
            self.node_visits += 1
            self.comparisons += 1
            if key == x.key:
                return x.value
            x = x.left if key < x.key else x.right
        return None

    def insert(self, key, value=None):
        """Returns True if a new key was added, False if an existing key was updated."""
        if value is None:
            value = key
        y, x = self.nil, self.root
        while x is not self.nil:
            y = x
            self.node_visits += 1
            self.comparisons += 1
            if key == x.key:
                x.value = value
                return False
            x = x.left if key < x.key else x.right
        z = _RBNode(key, value, RED)
        z.left = z.right = self.nil
        z.parent = y
        if y is self.nil:
            self.root = z
        elif key < y.key:
            y.left = z
        else:
            y.right = z
        self.size += 1
        self._fix_insert(z)
        return True

    def _fix_insert(self, z):
        while z.parent.color == RED:
            gp = z.parent.parent
            if z.parent is gp.left:
                uncle = gp.right
                if uncle.color == RED:
                    z.parent.color = uncle.color = BLACK
                    gp.color = RED
                    z = gp
                else:
                    if z is z.parent.right:
                        z = z.parent
                        self._rotate_left(z)
                    z.parent.color = BLACK
                    z.parent.parent.color = RED
                    self._rotate_right(z.parent.parent)
            else:
                uncle = gp.left
                if uncle.color == RED:
                    z.parent.color = uncle.color = BLACK
                    gp.color = RED
                    z = gp
                else:
                    if z is z.parent.left:
                        z = z.parent
                        self._rotate_right(z)
                    z.parent.color = BLACK
                    z.parent.parent.color = RED
                    self._rotate_left(z.parent.parent)
        self.root.color = BLACK

    def height(self):
        def h(x):
            return 0 if x is self.nil else 1 + max(h(x.left), h(x.right))
        return h(self.root)

    def inorder(self):
        out, stack, x = [], [], self.root
        while stack or x is not self.nil:
            while x is not self.nil:
                stack.append(x)
                x = x.left
            x = stack.pop()
            out.append(x.key)
            x = x.right
        return out

    def is_valid(self):
        """Checks BST order, black root, no red-red, equal black height."""
        if self.root.color != BLACK:
            return False

        def check(x, lo, hi):
            if x is self.nil:
                return 1
            if (lo is not None and x.key <= lo) or (hi is not None and x.key >= hi):
                return -1
            if x.color == RED and (x.left.color == RED or x.right.color == RED):
                return -1
            a, b = check(x.left, lo, x.key), check(x.right, x.key, hi)
            if a == -1 or b == -1 or a != b:
                return -1
            return a + (1 if x.color == BLACK else 0)
        return check(self.root, None, None) != -1


# ----------------------------------------------------------------------------
# B-Tree of minimum degree t (CLRS, single-pass insert with preemptive split).
# search/insert O(t log_t n) CPU, O(log_t n) node (page) accesses. Space O(n).
# ----------------------------------------------------------------------------
class _BNode:
    __slots__ = ("keys", "values", "children", "leaf")

    def __init__(self, leaf):
        self.keys, self.values, self.children, self.leaf = [], [], [], leaf


class BTree:
    def __init__(self, t=16):
        if t < 2:
            raise ValueError("B-Tree minimum degree t must be >= 2.")
        self.t = t
        self.root = _BNode(True)
        self.size = 0
        self.splits = 0
        self.comparisons = 0
        self.node_visits = 0      # proxy for disk-page accesses

    def _locate(self, key):
        x = self.root
        while True:
            i = 0
            while i < len(x.keys) and key > x.keys[i]:
                i += 1
            if i < len(x.keys) and x.keys[i] == key:
                return x, i
            if x.leaf:
                return None
            x = x.children[i]

    def search(self, key):
        x = self.root
        while True:
            self.node_visits += 1
            i = 0
            while i < len(x.keys) and key > x.keys[i]:
                i += 1
                self.comparisons += 1
            self.comparisons += 1
            if i < len(x.keys) and x.keys[i] == key:
                return x.values[i]
            if x.leaf:
                return None
            x = x.children[i]

    def _split_child(self, x, i):
        t = self.t
        y = x.children[i]
        z = _BNode(y.leaf)
        mid_k, mid_v = y.keys[t - 1], y.values[t - 1]
        z.keys, z.values = y.keys[t:], y.values[t:]
        y.keys, y.values = y.keys[:t - 1], y.values[:t - 1]
        if not y.leaf:
            z.children, y.children = y.children[t:], y.children[:t]
        x.children.insert(i + 1, z)
        x.keys.insert(i, mid_k)
        x.values.insert(i, mid_v)
        self.splits += 1

    def insert(self, key, value=None):
        if value is None:
            value = key
        found = self._locate(key)
        if found:
            node, i = found
            node.values[i] = value
            return False
        if len(self.root.keys) == 2 * self.t - 1:
            s = _BNode(False)
            s.children = [self.root]
            self.root = s
            self._split_child(s, 0)
        x = self.root
        while True:
            self.node_visits += 1
            i = len(x.keys)
            while i > 0 and key < x.keys[i - 1]:
                i -= 1
                self.comparisons += 1
            self.comparisons += 1
            if x.leaf:
                x.keys.insert(i, key)
                x.values.insert(i, value)
                break
            if len(x.children[i].keys) == 2 * self.t - 1:
                self._split_child(x, i)
                self.comparisons += 1
                if key > x.keys[i]:
                    i += 1
            x = x.children[i]
        self.size += 1
        return True

    def height(self):
        h, x = 1, self.root
        while not x.leaf:
            x = x.children[0]
            h += 1
        return h

    def inorder(self):
        out = []

        def walk(x):
            for i, k in enumerate(x.keys):
                if not x.leaf:
                    walk(x.children[i])
                out.append(k)
            if not x.leaf:
                walk(x.children[-1])
        walk(self.root)
        return out

    def is_valid(self):
        depths = set()

        def check(x, depth, is_root):
            if len(x.keys) > 2 * self.t - 1:
                return False
            if not is_root and len(x.keys) < self.t - 1:
                return False
            if x.keys != sorted(x.keys):
                return False
            if x.leaf:
                depths.add(depth)
                return True
            if len(x.children) != len(x.keys) + 1:
                return False
            return all(check(c, depth + 1, False) for c in x.children)
        return check(self.root, 0, True) and len(depths) <= 1 and \
            self.inorder() == sorted(self.inorder())


# ----------------------------------------------------------------------------
# Binomial Heap.  insert/merge/extract-min/decrease-key all O(log n) worst case.
# ----------------------------------------------------------------------------
class _BinNode:
    __slots__ = ("key", "item", "children", "parent")

    def __init__(self, key, item):
        self.key, self.item, self.children, self.parent = key, item, [], None


class BinomialHeap:
    def __init__(self):
        self.roots = []
        self.size = 0
        self.ops = 0           # elementary steps: root visits + links + swaps
        self.handles = {}

    def _link(self, a, b):
        if b.key < a.key:
            a, b = b, a
        b.parent = a
        a.children.append(b)
        self.ops += 1
        return a

    def _consolidate(self, roots):
        by_deg = {}
        for r in roots:
            self.ops += 1
            while len(r.children) in by_deg:
                r = self._link(r, by_deg.pop(len(r.children)))
            by_deg[len(r.children)] = r
        self.roots = [by_deg[d] for d in sorted(by_deg)]

    def insert(self, key, item):
        node = _BinNode(key, item)
        self.handles[item] = node
        self._consolidate(self.roots + [node])
        self.size += 1

    def find_min(self):
        if not self.roots:
            raise IndexError("heap is empty")
        self.ops += len(self.roots)
        m = min(self.roots, key=lambda r: r.key)
        return m.key, m.item

    def extract_min(self):
        if not self.roots:
            raise IndexError("heap is empty")
        self.ops += len(self.roots)
        m = min(self.roots, key=lambda r: r.key)
        self.roots.remove(m)
        for c in m.children:
            c.parent = None
        self._consolidate(self.roots + m.children)
        self.size -= 1
        del self.handles[m.item]
        return m.key, m.item

    def decrease_key(self, item, new_key):
        node = self.handles[item]
        if new_key > node.key:
            raise ValueError("new key is greater than current key")
        node.key = new_key
        while node.parent is not None and node.key < node.parent.key:
            p = node.parent
            node.key, p.key = p.key, node.key
            node.item, p.item = p.item, node.item
            self.handles[node.item], self.handles[p.item] = node, p
            node = p
            self.ops += 1

    def merge(self, other):
        self._consolidate(self.roots + other.roots)
        self.size += other.size
        self.handles.update(other.handles)
        other.roots, other.size, other.handles = [], 0, {}

    def __len__(self):
        return self.size


# ----------------------------------------------------------------------------
# Fibonacci Heap.  insert/merge/find-min/decrease-key O(1) amortized,
# extract-min O(log n) amortized.  Larger constant factors than binomial.
# ----------------------------------------------------------------------------
class _FibNode:
    __slots__ = ("key", "item", "parent", "child", "left", "right", "degree", "mark")

    def __init__(self, key, item):
        self.key, self.item = key, item
        self.parent = self.child = None
        self.left = self.right = self
        self.degree = 0
        self.mark = False


def _ring_remove(x):
    x.left.right = x.right
    x.right.left = x.left
    x.left = x.right = x


def _ring_add(head, x):
    """Insert x to the right of head (head's ring)."""
    x.left = head
    x.right = head.right
    head.right.left = x
    head.right = x


class FibonacciHeap:
    def __init__(self):
        self.min = None
        self.size = 0
        self.ops = 0           # elementary steps: root visits + links + cuts
        self.links = 0
        self.cuts = 0
        self.handles = {}

    def _add_root(self, x):
        x.parent = None
        if self.min is None:
            x.left = x.right = x
            self.min = x
        else:
            _ring_add(self.min, x)
            if x.key < self.min.key:
                self.min = x

    def insert(self, key, item):
        node = _FibNode(key, item)
        self.handles[item] = node
        self._add_root(node)
        self.size += 1
        self.ops += 1

    def find_min(self):
        if self.min is None:
            raise IndexError("heap is empty")
        return self.min.key, self.min.item

    def merge(self, other):
        self.handles.update(other.handles)
        if other.min is not None:
            if self.min is None:
                self.min = other.min
            else:
                a, b = self.min, other.min
                a_r, b_l = a.right, b.left
                a.right, b.left = b, a
                b_l.right, a_r.left = a_r, b_l
                if b.key < a.key:
                    self.min = b
        self.size += other.size
        self.ops += 1
        other.min, other.size, other.handles = None, 0, {}

    def extract_min(self):
        z = self.min
        if z is None:
            raise IndexError("heap is empty")
        children = []
        if z.child is not None:
            c = z.child
            while True:
                children.append(c)
                c = c.right
                if c is z.child:
                    break
        if z.right is z:
            self.min = None
        else:
            self.min = z.right
            _ring_remove(z)
        for c in children:
            c.left = c.right = c
            self._add_root(c)
        z.child = None
        self.size -= 1
        del self.handles[z.item]
        if self.min is not None:
            self._consolidate()
        return z.key, z.item

    def _consolidate(self):
        roots, x = [], self.min
        while True:
            roots.append(x)
            x = x.right
            if x is self.min:
                break
        self.min = None
        for r in roots:
            r.left = r.right = r
        table = {}
        for x in roots:
            self.ops += 1
            d = x.degree
            while d in table:
                y = table.pop(d)
                if y.key < x.key:
                    x, y = y, x
                self._link(y, x)
                d += 1
            table[d] = x
        for x in table.values():
            self._add_root(x)

    def _link(self, y, x):
        y.parent = x
        y.mark = False
        if x.child is None:
            x.child = y
            y.left = y.right = y
        else:
            _ring_add(x.child, y)
        x.degree += 1
        self.links += 1
        self.ops += 1

    def _cut(self, x, y):
        if y.child is x:
            y.child = x.right if x.right is not x else None
        _ring_remove(x)
        y.degree -= 1
        x.mark = False
        self._add_root(x)
        self.cuts += 1
        self.ops += 1

    def _cascading_cut(self, y):
        z = y.parent
        while z is not None:
            if not y.mark:
                y.mark = True
                return
            self._cut(y, z)
            y, z = z, z.parent

    def decrease_key(self, item, new_key):
        x = self.handles[item]
        if new_key > x.key:
            raise ValueError("new key is greater than current key")
        x.key = new_key
        y = x.parent
        if y is not None and x.key < y.key:
            self._cut(x, y)
            self._cascading_cut(y)
        if x.key < self.min.key:
            self.min = x
        self.ops += 1

    def __len__(self):
        return self.size


# ----------------------------------------------------------------------------
# Workloads used by the framework and the experiments
# ----------------------------------------------------------------------------
def tree_workload(n, searches_per_insert=5, btree_t=16, seed=11):
    """Insert n distinct sensor-record keys, then run n*ratio point searches."""
    rnd = random.Random(seed)
    keys = rnd.sample(range(10 * n), n)
    queries = [rnd.choice(keys) for _ in range(int(n * searches_per_insert))]
    rows = []
    for name, tree in (("Red-Black Tree", RedBlackTree()), ("B-Tree (t=%d)" % btree_t, BTree(btree_t))):
        t0 = time.perf_counter()
        for k in keys:
            tree.insert(k)
        ins_ms = (time.perf_counter() - t0) * 1000
        ins_visits, ins_cmp = tree.node_visits, tree.comparisons
        tree.node_visits = tree.comparisons = 0
        t0 = time.perf_counter()
        hits = sum(1 for q in queries if tree.search(q) is not None)
        srch_ms = (time.perf_counter() - t0) * 1000
        rebalance = tree.rotations if isinstance(tree, RedBlackTree) else tree.splits
        rows.append({
            "structure": name, "n": n, "height": tree.height(),
            "insert_ms": round(ins_ms, 3), "search_ms": round(srch_ms, 3),
            "insert_node_visits": ins_visits, "insert_comparisons": ins_cmp,
            "search_node_visits": tree.node_visits, "search_comparisons": tree.comparisons,
            "rebalance_ops": rebalance,
            "rebalance_kind": "rotations" if isinstance(tree, RedBlackTree) else "node splits",
            "valid": tree.is_valid(), "hits": hits, "queries": len(queries),
        })
    return rows


def heap_workload(n, decrease_per_item=1.0, seed=13):
    """Insert n events, apply n*ratio decrease-key (priority escalations),
    extract all n. Also measures merging two heaps of n/2 events."""
    rnd = random.Random(seed)
    base = [rnd.randint(n, 10 * n) for _ in range(n)]
    dec = [(rnd.randrange(n), rnd.randint(1, n)) for _ in range(int(n * decrease_per_item))]
    rows = []
    for name, cls in (("Binomial Heap", BinomialHeap), ("Fibonacci Heap", FibonacciHeap)):
        ms = float("inf")
        for _rep in range(2):                         # ops are deterministic; keep the faster time
            h = cls()
            t0 = time.perf_counter()
            for i, k in enumerate(base):
                h.insert(k, i)
            o_ins = h.ops
            cur = list(base)
            for i, d in dec:                           # only ever decrease keys
                nk = cur[i] - d
                if nk < cur[i]:
                    h.decrease_key(i, nk)
                    cur[i] = nk
            o_dec = h.ops - o_ins
            last, ok = None, True
            for _ in range(n):
                k, _item = h.extract_min()
                ok = ok and (last is None or k >= last)
                last = k
            o_ext = h.ops - o_ins - o_dec
            ms = min(ms, (time.perf_counter() - t0) * 1000)
        # merge experiment
        a, b = cls(), cls()
        for i in range(n // 2):
            a.insert(base[i], i)
        for i in range(n // 2, n):
            b.insert(base[i], i)
        before = a.ops
        a.merge(b)
        o_merge = a.ops - before
        rows.append({"structure": name, "n": n, "decrease_keys": len(dec),
                     "insert_ops": o_ins, "decrease_key_ops": o_dec, "extract_min_ops": o_ext,
                     "merge_ops": o_merge, "total_ops": o_ins + o_dec + o_ext,
                     "total_ms": round(ms, 3), "sorted_ok": ok})
    return rows
