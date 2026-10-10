"""Fractional (greedy) and 0/1 (dynamic programming) knapsack.
Items are tuples (weight, value, name)."""


def _validate(items, capacity):
    if capacity < 0:
        raise ValueError("Capacity must be non-negative.")
    for w, v, _ in items:
        if w <= 0 or v < 0:
            raise ValueError("Weights must be positive and values non-negative.")


def fractional_knapsack(items, capacity):
    """Greedy by value/weight. Optimal when items are divisible. O(n log n) time."""
    _validate(items, capacity)
    remaining = capacity
    sel = []
    value = 0.0
    for w, v, name in sorted(items, key=lambda x: x[1] / x[0], reverse=True):
        if remaining <= 0:
            break
        take = min(1.0, remaining / w)
        sel.append({"name": name, "fraction": round(take, 4),
                    "weight_taken": round(w * take, 4), "value_added": round(v * take, 4)})
        remaining -= w * take
        value += v * take
    return {"selected": sel, "total_value": round(value, 4),
            "capacity_used": round(capacity - remaining, 4),
            "ops": len(items)}


def zero_one_knapsack(items, capacity):
    """Exact DP. O(nW) time and space -- pseudo-polynomial (W is a numeric value,
    not the bit-length of the input). Needs integer weights/capacity."""
    _validate(items, capacity)
    W = int(capacity)
    if abs(W - capacity) > 1e-9:
        raise ValueError("0/1 Knapsack DP needs an integer capacity.")
    if W > 20000:
        raise ValueError("Capacity too large for the demo (W <= 20000).")
    n = len(items)
    dp = [[0.0] * (W + 1) for _ in range(n + 1)]
    ops = 0
    for i, (w, v, _) in enumerate(items, 1):
        wi = int(w)
        if abs(wi - w) > 1e-9:
            raise ValueError("0/1 Knapsack DP needs integer item weights.")
        for c in range(W + 1):
            ops += 1
            dp[i][c] = dp[i - 1][c]
            if wi <= c and dp[i - 1][c - wi] + v > dp[i][c]:
                dp[i][c] = dp[i - 1][c - wi] + v
    sel, c = [], W
    for i in range(n, 0, -1):
        if dp[i][c] != dp[i - 1][c]:
            w, v, name = items[i - 1]
            sel.append({"name": name, "weight": int(w), "value": v})
            c -= int(w)
    sel.reverse()
    return {"selected": sel, "total_value": round(dp[n][W], 4),
            "capacity_used": sum(s["weight"] for s in sel),
            "ops": ops, "table_cells": (n + 1) * (W + 1)}


def greedy_zero_one(items, capacity):
    """Value/weight greedy applied to INDIVISIBLE items. Fast but NOT optimal --
    included only to show why 0/1 Knapsack needs a different technique."""
    _validate(items, capacity)
    remaining, value, sel = capacity, 0.0, []
    for w, v, name in sorted(items, key=lambda x: x[1] / x[0], reverse=True):
        if w <= remaining:
            sel.append(name)
            remaining -= w
            value += v
    return {"selected": sel, "total_value": round(value, 4)}
