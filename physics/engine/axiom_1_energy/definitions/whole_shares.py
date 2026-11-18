"""Whole shares: a whole number of quanta dealt over boxes in proportion, exactly.

Contract: `whole_shares(total, weights)` (host) returns a list of whole numbers, one per
weight, that sum to `total` exactly. Each gets the whole part of `total * weight / sum of
weights`; the quanta left over go one each to the largest remainders, a tie to the
lowest index. All weights 0: dealt evenly, the rest one each from index 0 up. Worked in
Python's whole numbers, so nothing is ever rounded. `total` and every weight must be
whole and 0 or more, and there must be a weight when there is anything to deal;
otherwise ValueError. When the weights are what each box holds and `total` is no more
than their sum, no box is asked for more than it holds.
"""
from __future__ import annotations


def whole_shares(total: int, weights) -> list:
    """`total` whole quanta over the weights in proportion, by largest remainders."""
    whole = [int(weight) for weight in weights]
    if int(total) != total or total < 0 or any(mine != given or mine < 0 for mine, given in zip(whole, weights)):
        raise ValueError("whole_shares deals a whole number, 0 or more, over whole weights, 0 or more")
    total = int(total)
    if not whole:
        if total:
            raise ValueError(f"no box to deal {total} quanta over")
        return []
    summed = sum(whole)
    if summed == 0:
        whole, summed = [1] * len(whole), len(whole)
    shares = [total * weight // summed for weight in whole]
    left = total - sum(shares)
    by_remainder = sorted(range(len(whole)), key=lambda index: (-(total * whole[index] % summed), index))
    for index in by_remainder[:left]:
        shares[index] += 1
    return shares
