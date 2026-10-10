import time


def timed(fn, *args, repeat=3, **kwargs):
    """Run fn repeat times; return (last_result, best_time_ms)."""
    best, result = float("inf"), None
    for _ in range(repeat):
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        best = min(best, (time.perf_counter() - t0) * 1000)
    return result, best
