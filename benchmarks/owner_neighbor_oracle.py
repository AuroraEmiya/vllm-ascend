"""CPU model of cached score boundaries and exact integer tie selection."""

import argparse
import bisect
import json
import math
import random


def select_prefix(peers, k, owner, trace=None):
    peers = [[(s, i) for s, i in p if i >= 0 and math.isfinite(s)] for p in peers]
    lo, hi = [0] * len(peers), [len(p) for p in peers]
    if sum(hi) <= k:
        return len(peers[owner])
    score_rounds = id_rounds = 0
    while True:
        widths = [h - begin for begin, h in zip(lo, hi)]
        residual = k - sum(lo)
        if residual <= len(peers):
            pivot = max(p[begin][0] for p, begin, h in zip(peers, lo, hi) if begin < h)
        elif sum(widths) - residual < len(peers):
            pivot = min(p[h - 1][0] for p, begin, h in zip(peers, lo, hi) if begin < h)
        else:
            rank = max(range(len(peers)), key=lambda r: widths[r])
            mid = lo[rank] + widths[rank] // 2
            if score_rounds == 0:
                mid = min(hi[rank] - 1, k // len(peers))
            pivot = peers[rank][mid][0]
        a, b = [], []
        for p, begin, h in zip(peers, lo, hi):
            neg = [-s for s, _ in p]
            a.append(bisect.bisect_left(neg, -pivot, begin, h))
            b.append(bisect.bisect_right(neg, -pivot, begin, h))
        score_rounds += 1
        greater, through = sum(a), sum(b)
        if greater < k <= through:
            break
        if k <= greater:
            hi = a
        else:
            lo = b
        assert score_rounds <= len(peers) * (k.bit_length() + 2)
    if through == k:
        result = b[owner]
    else:
        lo, hi = a, b
        while True:
            widths = [h - begin for begin, h in zip(lo, hi)]
            rank = max(range(len(peers)), key=lambda r: widths[r])
            mid = lo[rank] + widths[rank] // 2
            pivot_id = peers[rank][mid][1]
            a, b = [], []
            for p, begin, h in zip(peers, lo, hi):
                ids = [i for _, i in p]
                a.append(bisect.bisect_left(ids, pivot_id, begin, h))
                b.append(bisect.bisect_right(ids, pivot_id, begin, h))
            id_rounds += 1
            if sum(a) < k <= sum(b):
                result = b[owner]
                break
            if k <= sum(a):
                hi = a
            else:
                lo = b
            assert id_rounds <= len(peers) * (k.bit_length() + 2)
    if trace is not None:
        trace.append((score_rounds, id_rounds))
    return result


def select_prefix_owner(peers, k, owner, trace=None):
    peers = [[(s, i) for s, i in p if i >= 0 and math.isfinite(s)] for p in peers]
    lower, upper = [0] * len(peers), [len(p) for p in peers]
    if not peers[owner] or sum(upper) <= k:
        return upper[owner]
    tail_score, tail_id = peers[owner][-1]
    heads = [p[0] for rank, p in enumerate(peers) if rank != owner and p]
    if not any(s > tail_score or (s == tail_score and idx < tail_id) for s, idx in heads):
        return upper[owner]
    skewed = max(s for s, _ in heads) - peers[owner][0][0] > peers[owner][0][0] - tail_score
    score_rounds = id_rounds = 0
    found = False
    while lower[owner] < upper[owner] and not found:
        mid = (lower[owner] + upper[owner]) // 2
        if score_rounds == 0:
            mid = 0 if skewed else min(upper[owner] - 1, k // len(peers))
        pivot = peers[owner][mid][0]
        a, b = [], []
        for p, begin, end in zip(peers, lower, upper):
            values = [-s for s, _ in p]
            a.append(bisect.bisect_left(values, -pivot, begin, end))
            b.append(bisect.bisect_right(values, -pivot, begin, end))
        score_rounds += 1
        found = sum(a) < k <= sum(b)
        if sum(a) >= k:
            upper = a
        if sum(b) < k:
            lower = b
    result = lower[owner]
    if found:
        result = b[owner]
        if sum(b) > k and a[owner] < b[owner]:
            lower, upper = a, b
            found = False
            while lower[owner] < upper[owner] and not found:
                mid = (lower[owner] + upper[owner]) // 2
                pivot = peers[owner][mid][1]
                a, b = [], []
                for p, begin, end in zip(peers, lower, upper):
                    values = [idx for _, idx in p]
                    a.append(bisect.bisect_left(values, pivot, begin, end))
                    b.append(bisect.bisect_right(values, pivot, begin, end))
                id_rounds += 1
                found = sum(a) < k <= sum(b)
                if sum(a) >= k:
                    upper = a
                if sum(b) < k:
                    lower = b
            result = b[owner] if found else lower[owner]
    if trace is not None:
        trace.append((score_rounds, id_rounds))
    return result


def check_tile_partition(rng):
    for _ in range(5000):
        values = sorted(rng.randrange(33) for _ in range(rng.randrange(1, 256)))
        begin = rng.randrange(len(values) + 1)
        end = rng.randrange(begin, len(values) + 1)
        pivot = rng.randrange(33)
        tile = rng.choice([1, 4, 16, 64])
        for inclusive in [False, True]:
            lo, hi = 0, (end - begin + tile - 1) // tile
            while lo < hi:
                mid = (lo + hi) // 2
                value = values[min(begin + (mid + 1) * tile - 1, end - 1)]
                before = value <= pivot if inclusive else value < pivot
                if before:
                    lo = mid + 1
                else:
                    hi = mid
            start = min(begin + lo * tile, end)
            actual = start + sum(v <= pivot if inclusive else v < pivot for v in values[start : min(start + tile, end)])
            expected = (bisect.bisect_right if inclusive else bisect.bisect_left)(values, pivot, begin, end)
            assert actual == expected


def selftest(seed=20878, cases=3000, directed=False):
    rng = random.Random(seed)
    trace = []
    for case in range(cases):
        world = rng.choice([1, 2, 3, 4, 8, 16, 17])
        k = rng.choice([1, 2, 4, 16, 32, 64, 128])
        interleave = rng.choice([1, 3, 768])
        peers = []
        for rank in range(world):
            p = []
            for local in range(rng.randrange(k + 1)):
                idx = local + ((local // interleave) * (world - 1) + rank) * interleave
                idx += ((2**24 // (world * interleave) + 17) * world * interleave) if case % 13 == 0 else 0
                score = rng.gauss(0, 1)
                if case % 3 == 0:
                    score = round(score)
                if case % 7 == 0:
                    score = rng.choice([0.0, -0.0])
                if case % 11 == 0:
                    score += 20 * (rank == world - 1)
                p.append((score, idx))
            p.sort(key=lambda pair: (-pair[0], pair[1]))
            peers.append(p + [(-math.inf, -1)] * (k - len(p)))
        all_pairs = sorted((p for peer in peers for p in peer if p[1] >= 0), key=lambda p: (-p[0], p[1]))[:k]
        for owner in {0, world - 1, rng.randrange(world)}:
            selected = (select_prefix_owner if directed else select_prefix)(peers, k, owner, trace)
            expected = sum((idx // interleave) % world == owner for _, idx in all_pairs)
            assert selected == expected, (case, k, world, owner, selected, expected)
    if directed:
        check_tile_partition(rng)
    return dict(
        status="passed",
        directed=directed,
        seed=seed,
        cases=cases,
        selections=len(trace),
        max_score_rounds=max(x[0] for x in trace),
        max_id_rounds=max(x[1] for x in trace),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=int, default=3000)
    parser.add_argument("--directed", action="store_true")
    args = parser.parse_args()
    print(json.dumps(selftest(cases=args.cases, directed=args.directed)))
