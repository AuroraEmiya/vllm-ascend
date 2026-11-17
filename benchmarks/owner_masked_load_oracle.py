# SPDX-License-Identifier: Apache-2.0
"""Field-level zero-fill/mask properties, reusing N4's independent sort oracle."""

import argparse
import json

import owner_contiguous_cache_oracle as oracle

original_count_window = oracle.count_window
mask_stats = dict(tiles=0, physical_pair_slots=0, logically_loaded_pairs=0, zero_filled_pairs=0, clipped_tiles=0)


def masked_count_window(peers, owner, pivot, lower, upper, tile):
    k = len(peers[0])
    counts = list(lower)
    active = [r != owner and lo < hi for r, (lo, hi) in enumerate(zip(lower, upper))]
    start = min((lo for lo, enabled in zip(lower, active) if enabled), default=k)
    tiles = 0
    while start < k and any(active):
        scan_start = min(start, k - tile)
        mask_stats["clipped_tiles"] += scan_start != start
        for rank, enabled in enumerate(active):
            amount = 0
            for lane in range(tile):
                position = scan_start + lane
                assert 0 <= position < k
                valid_region = enabled and max(lower[rank], start) <= position < upper[rank]
                pair_mask = [
                    enabled
                    and max(lower[rank], start) <= scan_start + field_lane // 2 < upper[rank]
                    and 0 <= scan_start + field_lane // 2 < k
                    for field_lane in (2 * lane, 2 * lane + 1)
                ]
                assert pair_mask[0] == pair_mask[1] == valid_region
                pair = peers[rank][position] if pair_mask[0] else (0.0, 0)
                # Zero-filled IDs/scores would look valid without this retained predicate.
                amount += valid_region and oracle.better(pair, pivot)
                mask_stats["physical_pair_slots"] += 1
                mask_stats["logically_loaded_pairs"] += pair_mask[0]
                mask_stats["zero_filled_pairs"] += not pair_mask[0]
            counts[rank] += amount
            capacity = max(0, min(start + tile, upper[rank]) - max(start, lower[rank]))
            active[rank] = enabled and amount == capacity and start + tile < upper[rank]
        tiles += 1
        start += tile
    expected, expected_tiles = original_count_window(peers, owner, pivot, lower, upper, tile)
    assert counts == expected and tiles == expected_tiles
    mask_stats["tiles"] += tiles
    return counts, tiles


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=int, default=4000)
    args = parser.parse_args()
    oracle.count_window = masked_count_window
    result = oracle.selftest(args.cases)
    result["mask_properties"] = mask_stats
    result["caveat"] = "Logical masks and zero-fill semantics only; physical NPU traffic remains unmeasured."
    print(json.dumps(result))
