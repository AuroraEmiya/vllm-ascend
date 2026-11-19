# SPDX-License-Identifier: Apache-2.0
"""Synthetic owner-prefix probes on either side of the skew routing threshold."""

import torch


def make_inputs(rows, world, owner, interleave, distribution, edge=False):
    assert world > 1 and not edge
    k = 2048
    offsets = torch.arange(k)
    ranks = torch.arange(world)[:, None]
    ids = (offsets // interleave * world + ranks) * interleave + offsets % interleave
    scores = torch.linspace(-2.0, -3.0, k).repeat(world, 1)
    scores[owner] = torch.linspace(1.0, 0.0, k)
    peer = (owner + 1) % world
    if distribution in ("deep_parallel", "deep_serial"):
        scores[peer] = torch.linspace(1.5, 1.1, k)
        # All global winners belong to peer on either side of the threshold.
        scores[peer, 0] = 1.99 if distribution == "deep_parallel" else 2.01
    else:
        raise ValueError(distribution)
    pair = torch.stack((ids.int(), scores.view(torch.int32)), dim=-1)
    packed = pair[:, None].expand(world, rows, k, 2).contiguous().view(world * rows, k, 2)
    # The unique top K scores are all from peer by construction.
    expected = torch.full((rows, 1, k), -1, dtype=torch.int32)
    return packed, expected
