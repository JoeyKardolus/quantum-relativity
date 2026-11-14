"""Pull-fade plot for Grid dynamics.

Contract: `plot_pull_fade(profile, out_dir)` draws the extra pull a packed lump
adds at each hop count (from `pull_by_hops`), as a share of the background
pull, on a log axis, writes pull_fade.png into out_dir and returns its name.
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def plot_pull_fade(profile: dict, out_dir: str) -> str:
    """Write pull_fade.png; return its file name."""
    figure, axes = plt.subplots(figsize=(6.5, 4.2), dpi=110)
    share = np.array(profile["excess"]) / profile["background"]
    axes.plot(profile["hops"], share, "o-", label="extra pull from the lump")
    axes.axhline(1.0, color="grey", linestyle="--", label="background pull (no lump)")
    axes.set_yscale("log")
    axes.set_xlabel("connections between the joint and the lump centre (hops)")
    axes.set_ylabel("extra pull / background pull")
    axes.set_title("Rule 7: pull spreads through the connections and fades with distance")
    axes.grid(alpha=0.3)
    axes.legend(fontsize=8)
    figure.tight_layout()
    name = "pull_fade.png"
    figure.savefig(os.path.join(out_dir, name))
    plt.close(figure)
    return name
