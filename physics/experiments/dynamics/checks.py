"""Per-rule checks for Grid dynamics.

Contract: `rule_checks(seed)` switches on one rule at a time on a small seeded
network and returns (numbers, passed): flat {"name", "measured", "known",
"note"} entries and one bool per check. Known values come from the rule
definitions, not from the code: drift lengthens by (1 + h)^n, energy flow
conserves total energy, the pull shortens an energy-dense part more than a
sparse one and fades the more connections lie between (rule 7), joint push
never exceeds push_limit and equals joint_push x squeeze below it (rule 8),
joint collapse keeps all energy (rule 8), repulsion leaves uncharged
connections alone, rewiring keeps the connection count, the energy-length
rule restores energy * length = LENGTH_SCALE. Pass bands are fixed here,
before any run. `pull_profile` is the pull-fade profile the checks used.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from physics.network import LENGTH_SCALE, Network, random_network, spatial_reference
from physics.rules import RuleSettings, collapse, push, step

_JOINTS = 2000
_DRIFT = 0.01
_DRIFT_STEPS = 10


def rule_checks(seed: int, pull_profile: dict) -> tuple[list[dict], list[bool]]:
    """Run every rule check; return the number entries and whether each passed."""
    base = spatial_reference(_JOINTS, 3, seed=seed)[0]
    results = [_energy_length(base, seed), _drift(base), _flow(base), _attraction(seed),
               _pull_fades(pull_profile), _push_cap(seed), _collapse_keeps_energy(base, seed),
               _repulsion(base), _rewiring(seed)]
    numbers = [entry for entries, _ in results for entry in entries]
    return numbers, [passed for _, passed in results]


def _entry(name, measured, known, note) -> dict:
    return {"name": name, "measured": measured, "known": known, "note": note}


def _energy_length(base: Network, seed: int):
    rng = np.random.default_rng(seed)
    knocked = replace(base, energy=base.energy * rng.uniform(0.5, 2.0, base.connection_count))
    after = step(knocked, RuleSettings(energy_length=1.0))
    worst = float(np.abs(after.energy * after.length - LENGTH_SCALE).max())
    return [_entry("energy-length: worst |energy x length - 1| after one step from a knocked network",
                   worst, 0.0, "energies scrambled by x0.5..x2 first")], worst < 1e-9


def _drift(base: Network):
    after = base
    for _ in range(_DRIFT_STEPS):
        after = step(after, RuleSettings(drift=_DRIFT))
    ratio = float(after.length.mean() / base.length.mean())
    known = (1 + _DRIFT) ** _DRIFT_STEPS
    return [_entry(f"drift h={_DRIFT}: mean length ratio after {_DRIFT_STEPS} steps",
                   round(ratio, 6), round(known, 6), "(1 + h)^n")], abs(ratio - known) < 1e-9


def _flow(base: Network):
    after = step(base, RuleSettings(energy_flow=0.5, energy_length=0.0))
    total = float(after.energy.sum() / base.energy.sum())
    spread = float(after.energy.std() / base.energy.std())
    numbers = [_entry("energy flow: total energy after / before", round(total, 12), 1.0,
                      "flow conserves energy"),
               _entry("energy flow: energy spread after / before", round(spread, 4), None,
                      "below 1 means energy evened out")]
    return numbers, abs(total - 1) < 1e-12 and spread < 1


def _attraction(seed: int):
    network = random_network(500, mean_degree=6, seed=seed)
    dense = np.zeros(network.joint_count, bool)
    dense[:100] = True
    hot = dense[network.pairs[:, 0]] & dense[network.pairs[:, 1]]
    cold = ~dense[network.pairs[:, 0]] & ~dense[network.pairs[:, 1]]
    network = Network.from_lengths(network.joint_count, network.pairs,
                                   1.0 / np.where(hot, 3.0, 1.0))
    shrink = step(network, RuleSettings(attraction=0.05)).length / network.length
    numbers = [_entry("attraction: length ratio inside the energy-dense part", round(float(shrink[hot].mean()), 4),
                      None, "100 of 500 joints at 3x energy"),
               _entry("attraction: length ratio in the sparse part", round(float(shrink[cold].mean()), 4),
                      None, "should shrink less than the dense part")]
    return numbers, bool((shrink < 1).all() and shrink[hot].mean() < shrink[cold].mean())


def _pull_fades(profile: dict):
    excess = np.array(profile["excess"])
    numbers = [_entry("pull: extra pull at a packed lump's centre / background pull",
                      round(float(excess[0] / profile["background"]), 4), None,
                      "how far a lump stands out from the pull everywhere"),
               _entry(f"pull: extra pull {profile['hops'][4]} connections away / at the centre",
                      round(float(excess[4] / excess[0]), 4), None, "below 1 means the pull fades"),
               _entry(f"pull: extra pull {profile['hops'][-1]} connections away / at the centre",
                      round(float(excess[-1] / excess[0]), 4), None, "keeps falling with every connection")]
    return numbers, bool(excess[0] > 0 and np.all(np.diff(excess) < 0))


def _push_cap(seed: int):
    squeeze = np.random.default_rng(seed).uniform(0.0, 2.0, 1000)
    settings = RuleSettings(joint_push=0.8, push_limit=1.0)
    pushed = push(squeeze, settings)
    below = settings.joint_push * squeeze < settings.push_limit
    numbers = [_entry("joint push: largest push / push_limit", float(pushed.max() / settings.push_limit), 1.0,
                      f"{int((~below).sum())} of 1000 joints asked for more than the limit"),
               _entry("joint push: largest |push - joint_push x squeeze| below the limit",
                      float(np.abs(pushed[below] - settings.joint_push * squeeze[below]).max()), 0.0,
                      "under the limit a joint pushes back in proportion to its squeeze")]
    return numbers, bool(pushed.max() == settings.push_limit and numbers[1]["measured"] < 1e-12)


def _collapse_keeps_energy(base: Network, seed: int):
    giving_way = np.random.default_rng(seed).random(base.joint_count) < 0.05
    after = collapse(base, giving_way)
    total = float(after.energy.sum() / base.energy.sum())
    numbers = [_entry("joint collapse: total energy after / before", round(total, 12), 1.0,
                      f"{int(giving_way.sum())} of {base.joint_count} joints gave way"),
               _entry("joint collapse: joints after / before", after.joint_count / base.joint_count, None,
                      "below 1: joints merged into a neighbour")]
    return numbers, abs(total - 1) < 1e-12 and after.joint_count < base.joint_count

def _repulsion(base: Network):
    charge = np.zeros(base.joint_count)
    charge[0] = 1.0
    charged = replace(base, charge=charge)
    after = step(charged, RuleSettings(repulsion=0.1))
    touched = (base.pairs == 0).any(axis=1)
    stretch = float((after.length[touched] / base.length[touched]).mean())
    others = float(np.abs(after.length[~touched] / base.length[~touched] - 1).max())
    numbers = [_entry("repulsion: charged joint's connections, length after / before", round(stretch, 4),
                      None, "above 1 means pushed apart"),
               _entry("repulsion: largest change on uncharged connections", others, 0.0,
                      "nothing else should move")]
    return numbers, stretch > 1 and others < 1e-12


def _rewiring(seed: int):
    network = random_network(800, mean_degree=6, seed=seed)
    settings = RuleSettings(rewiring=5000)
    after = step(network, settings, np.random.default_rng(seed))
    before_action, after_action = _action(network, settings), _action(after, settings)
    numbers = [_entry("rewiring: connection count after / before",
                      after.connection_count / network.connection_count, 1.0, "moves, never adds"),
               _entry("rewiring: action after / before", round(after_action / before_action, 4), None,
                      "below 1 means the moves lowered the action")]
    return numbers, after.connection_count == network.connection_count and after_action < before_action


def _action(network: Network, settings: RuleSettings) -> float:
    adjacency = network.adjacency(np.ones(network.connection_count))
    triangles = (adjacency @ adjacency).multiply(adjacency).sum(axis=1).A.ravel() / 2
    degree_term = ((network.degree() - settings.target_degree) ** 2).sum()
    triangle_term = ((triangles - settings.target_triangles) ** 2).sum()
    return float(degree_term + settings.triangle_weight * triangle_term)
