"""Ball-growth (Hausdorff) dimension: how joint counts grow with distance.

Contract: `ball_dimension(network, sources, seed, inner, outer)` picks
`sources` random joints, counts joints within distance r of each (shortest
paths weighted by length), averages, and fits log count = D log r + c on radii
from `inner` x the median connection length up to `outer` x the mean largest
distance. The two cut-offs are fixed defaults, never tuned per network: below
the inner one the count is grainy, above the outer one a finite network runs
out of room. Returns {"dimension", "radii", "counts", "fit_range"}.
"""
from __future__ import annotations

import numpy as np

from physics.network import Network
from physics.measure.distance import distances


def ball_dimension(network: Network, sources: int = 64, seed: int = 0,
                   inner: float = 3.0, outer: float = 0.4) -> dict:
    """Fitted exponent D of N(r) ~ r^D, from length-weighted shortest paths."""
    rng = np.random.default_rng(seed)
    chosen = rng.choice(network.joint_count, size=min(sources, network.joint_count),
                        replace=False)
    reach = distances(network, chosen)
    finite = np.where(np.isfinite(reach), reach, np.nan)
    low = inner * float(np.median(network.length))
    high = outer * float(np.nanmean(np.nanmax(finite, axis=1)))
    if not high > low:
        raise ValueError("network too small: no room between the inner and outer radius")
    radii = np.geomspace(low, high, 16)
    sorted_rows = np.sort(np.where(np.isfinite(reach), reach, np.inf), axis=1)
    counts = np.array([np.mean([np.searchsorted(row, radius, side="right")
                                for row in sorted_rows]) for radius in radii])
    slope, _ = np.polyfit(np.log(radii), np.log(counts), 1)
    return {"dimension": float(slope), "radii": radii, "counts": counts,
            "fit_range": (low, high)}
