# SPDX-License-Identifier: Apache-2.0
"""Independent global sort and cached-window properties for N4's normal-ID domain."""

import argparse
import bisect
import json
import math
import random
import struct


def f32(value):
    return struct.unpack("f", struct.pack("f", value))[0]


def better(pair, pivot):
    score, idx = pair
    return idx >= 0 and math.isfinite(score) and (-score, idx) < (-pivot[0], pivot[1])


def count_window(peers, owner, pivot, lower, upper, tile):
    """Mirror shared scalar tiles, including physical tail clipping, without Triton."""
    k = len(peers[0])
    counts = list(lower)
    active = [r != owner and lo < hi for r, (lo, hi) in enumerate(zip(lower, upper))]
    start = min((lo for lo, enabled in zip(lower, active) if enabled), default=k)
    tiles = 0
    while start < k and any(active):
        scan_start = min(start, k - tile)
        for rank, enabled in enumerate(active):
            if not enabled:
                continue
            capacity = max(0, min(start + tile, upper[rank]) - max(start, lower[rank]))
            amount = sum(
                better(peers[rank][position], pivot)
                for position in range(scan_start, scan_start + tile)
                if max(start, lower[rank]) <= position < upper[rank]
            )
            counts[rank] += amount
            active[rank] = amount == capacity and start + tile < upper[rank]
        tiles += 1
        start += tile
    expected = [sum(better(pair, pivot) for pair in peer) if rank != owner else 0 for rank, peer in enumerate(peers)]
    assert counts == expected, (lower, upper, counts, expected)
    return counts, tiles


def select(peers, owner, interleave, cached, random_pivots, rng, stats):
    k, world = len(peers[0]), len(peers)
    own = peers[owner]
    own_valid = [(s, idx) for s, idx in own if idx >= 0 and math.isfinite(s)]
    upper, lower = len(own_valid), 0
    if not upper:
        return 0
    head, tail = own_valid[0], own_valid[-1]
    if not any(rank != owner and better(peer[0], tail) for rank, peer in enumerate(peers)):
        lower = upper
    peak = max((peer[0][0] for rank, peer in enumerate(peers) if rank != owner and peer[0][1] >= 0), default=-math.inf)
    skewed = peak - head[0] > head[0] - tail[0]
    parallel = world > 4 and not skewed
    tile = min(k, 1 << max(0, (max(1, k // world) - 1).bit_length()))
    pivot = 0 if skewed else min(k // world, upper - 1)
    cache_lo, cache_hi = [0] * world, [k if r != owner else 0 for r in range(world)]
    iteration, direction, previous = 0, 0, None
    valid_keys = [[(-s, idx) for s, idx in peer if idx >= 0 and math.isfinite(s)] for peer in peers]
    limit = 2 * k + 4 if random_pivots else k.bit_length() - 1 + 4
    while lower < upper and iteration < limit:
        if random_pivots:
            pivot = rng.randrange(lower, upper)
        if previous is not None and pivot != previous:
            new_direction = 1 if pivot > previous else -1
            stats["direction_reversals"] += bool(direction and direction != new_direction)
            direction = new_direction
        previous = pivot
        pair = own[pivot]
        counts = [
            bisect.bisect_left(keys, (-pair[0], pair[1])) if rank != owner else 0
            for rank, keys in enumerate(valid_keys)
        ]
        if parallel:
            if cached:
                actual, tiles = count_window(peers, owner, pair, cache_lo, cache_hi, tile)
                assert actual == counts
                stats["cached_tiles"] += tiles
            # The baseline stops at the first partial tile, including the tile
            # containing the first non-better entry when c is a tile multiple.
            baseline_tiles = max(
                (min(k // tile, amount // tile + 1) for rank, amount in enumerate(counts) if rank != owner), default=0
            )
            if cached:
                stats["baseline_tiles_for_cached_queries"] += baseline_tiles
                assert tiles <= baseline_tiles, (tiles, baseline_tiles, cache_lo, cache_hi, counts)
        count = pivot + sum(counts)
        selected = count < k
        if parallel and cached:
            if selected:
                cache_lo = counts
            else:
                cache_hi = counts
        jump = k - count + pivot
        lower = pivot + 1 if selected else max(lower, jump)
        upper = min(upper, jump) if selected else pivot
        if head[0] == tail[0] and lower < upper:
            block_begin = pair[1] // interleave * interleave
            block_last = block_begin + interleave - 1
            block_start = sum(idx < block_begin for _, idx in own_valid)
            block_end = sum(idx <= block_last for _, idx in own_valid)
            if block_start <= jump <= block_end:
                lower = upper = jump
            else:
                if jump > block_end:
                    lower = max(lower, block_end)
                if jump < block_start:
                    upper = min(upper, block_start)
        if parallel and cached and lower < upper:
            for rank, keys in enumerate(valid_keys):
                if rank == owner:
                    continue
                for candidate in (lower, upper - 1):
                    exact = bisect.bisect_left(keys, (-own[candidate][0], own[candidate][1]))
                    assert cache_lo[rank] <= exact <= cache_hi[rank]
        if iteration < 2:
            estimate = math.trunc(pivot + (k - count) / world)
            pivot = min(max(estimate, lower), upper - 1)
        else:
            pivot = (lower + upper) // 2
        iteration += 1
    assert lower == upper, (k, world, owner, cached, random_pivots, lower, upper, iteration)
    stats["searches"] += 1
    stats["max_rounds"] = max(stats["max_rounds"], iteration)
    return lower


def selftest(cases=4000, seed=20878):
    rng = random.Random(seed)
    stats = dict(searches=0, max_rounds=0, direction_reversals=0, baseline_tiles_for_cached_queries=0, cached_tiles=0)
    for case in range(cases):
        world = rng.choice([1, 2, 3, 4, 8, 16, 17])
        k = rng.choice([1, 2, 4, 16, 32, 64, 128]) if case < cases - 60 else 2048
        interleave = rng.choice([1, 3, 768])
        peers = []
        for rank in range(world):
            valid = k if case % 5 else rng.randrange(k + 1)
            peer = []
            for local in range(valid):
                idx = local + ((local // interleave) * (world - 1) + rank) * interleave
                score = f32(rng.gauss(0, 1))
                if case % 3 == 0:
                    score = round(score)
                if case % 7 == 0:
                    score = rng.choice([0.0, -0.0])
                if case % 11 == 0:
                    score += 20 * (rank == world - 1)
                peer.append((score, idx))
            peer.sort(key=lambda pair: (-pair[0], pair[1]))
            peers.append(peer + [(-math.inf, -1)] * (k - valid))
        winners = sorted(
            (pair for peer in peers for pair in peer if pair[1] >= 0), key=lambda pair: (-pair[0], pair[1])
        )[:k]
        for owner in {0, world - 1, rng.randrange(world)}:
            expected = sum((idx // interleave) % world == owner for _, idx in winners)
            for cached, random_pivots in ((False, False), (True, False), (True, True)):
                actual = select(peers, owner, interleave, cached, random_pivots, rng, stats)
                assert actual == expected, (case, world, k, owner, cached, random_pivots, actual, expected)
    assert stats["direction_reversals"] > 0
    return dict(
        status="passed",
        cases=cases,
        seed=seed,
        **stats,
        caveat="Paired CPU shared-tile counts for the same cached queries; not actual NPU load traffic or timing.",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=int, default=4000)
    args = parser.parse_args()
    print(json.dumps(selftest(args.cases)))
