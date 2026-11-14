"""One step of the quantum-relativity theory rules, each switchable, none using coordinates.

Contract: `step(network, settings, rng)` returns a new Network, possibly with
fewer joints (joint collapse); its merged_into maps every old joint to its new
index. Order, each skipped when its strength is 0:
  1. energy flow   energy diffuses between connections that share a joint:
                   e_c += rate * (mean energy of c's neighbouring connections - e_c),
                   done symmetrically so total energy is conserved exactly.
  2. drift         every length *= (1 + drift), every energy /= (1 + drift),
                   the same everywhere (rule 6: lengthening is redshift).
  3. attraction    p = pull per joint from energy volume (pull.py, rule 7);
                   connection (a, b): length *= exp(-attraction * (p_a + p_b) / 2).
  4. joint push    squeeze_j = mean over j's connections of the log-shortening
                   attraction * (p_a + p_b) / 2 from step 3;
                   push_j = min(joint_push * squeeze_j, push_limit) (push.py);
                   connection (a, b): length *= exp((push_a + push_b) / 2) (rule 8).
  5. repulsion     a charged joint stretches its own connections:
                   length *= exp(repulsion * (|q_a| + |q_b|) / 2).
  6. energy-length energy and length move toward energy * length = LENGTH_SCALE,
                   splitting the correction evenly in log space; strength 1
                   restores the rule exactly. So a squeeze raises energy and a
                   stretch lowers it (rule 8: force, energy, length are one).
  7. joint collapse every joint with joint_push * squeeze_j > push_limit gives
                   way (collapse.py): it merges into the far end of its shortest
                   connection, keeping all energy.
  8. rewiring      `settings.rewiring` local move attempts (see rewire.py).
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from physics.network import LENGTH_SCALE, Network
from physics.rules.collapse import collapse
from physics.rules.pull import pull
from physics.rules.push import push
from physics.rules.rewire import rewire
from physics.rules.settings import RuleSettings


def step(network: Network, settings: RuleSettings = RuleSettings(),
         rng: np.random.Generator | None = None) -> Network:
    """Apply every switched-on rule once, in the documented order."""
    energy, length = network.energy.astype(float), network.length.astype(float)
    heads, tails = network.pairs[:, 0], network.pairs[:, 1]
    squeeze = np.zeros(network.joint_count)
    if settings.energy_flow:
        energy = _flow(network, energy, settings.energy_flow)
    if settings.drift:
        length = length * (1 + settings.drift)
        energy = energy / (1 + settings.drift)
    if settings.attraction:
        joint_pull = pull(network, energy, length)
        shortening = settings.attraction * (joint_pull[heads] + joint_pull[tails]) / 2
        length = length * np.exp(-shortening)
        squeeze = _joint_sums(network, shortening) / np.maximum(network.degree(), 1)
    asked = settings.joint_push * squeeze
    if settings.joint_push:
        pushed = push(squeeze, settings)
        length = length * np.exp((pushed[heads] + pushed[tails]) / 2)
    if settings.repulsion:
        charge = np.abs(network.charge)
        length = length * np.exp(settings.repulsion * (charge[heads] + charge[tails]) / 2)
    if settings.energy_length:
        mismatch = np.log(energy * length / LENGTH_SCALE)
        energy = energy * np.exp(-settings.energy_length * mismatch / 2)
        length = length * np.exp(-settings.energy_length * mismatch / 2)
    stepped = collapse(replace(network, energy=energy, length=length), asked > settings.push_limit)
    if settings.rewiring:
        stepped = rewire(stepped, settings, rng or np.random.default_rng())
    return stepped


def _joint_sums(network: Network, values: np.ndarray) -> np.ndarray:
    return np.bincount(network.pairs.ravel(), weights=np.repeat(values, 2),
                       minlength=network.joint_count)


def _flow(network: Network, energy: np.ndarray, rate: float) -> np.ndarray:
    """Symmetric diffusion on the connection-adjacency graph; conserves the total."""
    heads, tails = network.pairs[:, 0], network.pairs[:, 1]
    degree = network.degree()
    sums = _joint_sums(network, energy)
    neighbour_sum = sums[heads] + sums[tails] - 2 * energy
    neighbour_count = degree[heads] + degree[tails] - 2
    largest = max(int(neighbour_count.max(initial=0)), 1)
    return energy + rate / largest * (neighbour_sum - neighbour_count * energy)
