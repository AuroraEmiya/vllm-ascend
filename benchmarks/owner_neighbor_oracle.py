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
        widths = [h - l for l, h in zip(lo, hi)]
        residual = k - sum(lo)
        if residual <= len(peers):
            pivot = max(p[l][0] for p, l, h in zip(peers, lo, hi) if l < h)
        elif sum(widths) - residual < len(peers):
            pivot = min(p[h - 1][0] for p, l, h in zip(peers, lo, hi) if l < h)
        else:
            rank = max(range(len(peers)), key=lambda r: widths[r])
            mid = lo[rank] + widths[rank] // 2
            if score_rounds == 0:
                mid = min(hi[rank] - 1, k // len(peers))
            pivot = peers[rank][mid][0]
        a, b = [], []
        for p, l, h in zip(peers, lo, hi):
            neg = [-s for s, _ in p]
            a.append(bisect.bisect_left(neg, -pivot, l, h))
            b.append(bisect.bisect_right(neg, -pivot, l, h))
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
            widths = [h - l for l, h in zip(lo, hi)]
            rank = max(range(len(peers)), key=lambda r: widths[r])
            mid = lo[rank] + widths[rank] // 2
            pivot_id = peers[rank][mid][1]
            a, b = [], []
            for p, l, h in zip(peers, lo, hi):
                ids = [i for _, i in p]
                a.append(bisect.bisect_left(ids, pivot_id, l, h))
                b.append(bisect.bisect_right(ids, pivot_id, l, h))
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


def selftest(seed=20878, cases=3000):
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
                idx += (2**24 + 17) * world * interleave if case % 13 == 0 else 0
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
            selected = select_prefix(peers, k, owner, trace)
            expected = sum((idx // interleave) % world == owner for _, idx in all_pairs)
            assert selected == expected, (case, k, world, owner, selected, expected)
    return dict(status="passed", seed=seed, cases=cases, selections=len(trace), max_score_rounds=max(x[0] for x in trace), max_id_rounds=max(x[1] for x in trace))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=int, default=3000)
    args = parser.parse_args()
    print(json.dumps(selftest(cases=args.cases)))
