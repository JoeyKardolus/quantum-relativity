"""One step of the quantum-relativity theory rules, each switchable, none using coordinates.

Contract: `step(network, settings, rng)` returns a new Network. Order, each
skipped when its strength is 0:
  1. energy flow   energy diffuses between connections that share a joint:
                   e_c += rate * (mean energy of c's neighbouring connections - e_c),
                   done symmetrically so total energy is conserved exactly.
  2. drift         every length *= (1 + drift), every energy /= (1 + drift)
                   (rule 6: lengthening is redshift; this energy is lost).
  3. attraction    each joint's energy density rho_j = mean energy of its
                   connections; connection (a, b) shortens by length *=
                   exp(-attraction * (rho_a + rho_b) / 2). Denser regions pull
                   their connections shorter. Only neighbour sums are used.
  4. repulsion     a charged joint stretches its own connections:
                   length *= exp(repulsion * (|q_a| + |q_b|) / 2).
  5. energy-length energy and length move toward energy * length = LENGTH_SCALE,
                   splitting the correction evenly in log space; strength 1
                   restores the rule exactly. So a squeeze raises energy and a
                   stretch lowers it (rule 8: force, energy, length are one).
  6. rewiring      `settings.rewiring` local move attempts (see rewire.py).
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from physics.network import LENGTH_SCALE, Network
from physics.rules.settings import RuleSettings
from physics.rules.rewire import rewire


def step(network: Network, settings: RuleSettings = RuleSettings(),
         rng: np.random.Generator | None = None) -> Network:
    """Apply every switched-on rule once, in the documented order."""
    energy, length = network.energy.astype(float), network.length.astype(float)
    if settings.energy_flow:
        energy = _flow(network, energy, settings.energy_flow)
    if settings.drift:
        length = length * (1 + settings.drift)
        energy = energy / (1 + settings.drift)
    if settings.attraction:
        density = _joint_density(network, energy)
        pair_density = (density[network.pairs[:, 0]] + density[network.pairs[:, 1]]) / 2
        length = length * np.exp(-settings.attraction * pair_density)
    if settings.repulsion:
        charge = np.abs(network.charge)
        push = (charge[network.pairs[:, 0]] + charge[network.pairs[:, 1]]) / 2
        length = length * np.exp(settings.repulsion * push)
    if settings.energy_length:
        mismatch = np.log(energy * length / LENGTH_SCALE)
        energy = energy * np.exp(-settings.energy_length * mismatch / 2)
        length = length * np.exp(-settings.energy_length * mismatch / 2)
    stepped = replace(network, energy=energy, length=length)
    if settings.rewiring:
        stepped = rewire(stepped, settings, rng or np.random.default_rng())
    return stepped


def _joint_sums(network: Network, values: np.ndarray) -> np.ndarray:
    return np.bincount(network.pairs.ravel(), weights=np.repeat(values, 2),
                       minlength=network.joint_count)


def _joint_density(network: Network, energy: np.ndarray) -> np.ndarray:
    degree = np.maximum(network.degree(), 1)
    return _joint_sums(network, energy) / degree


def _flow(network: Network, energy: np.ndarray, rate: float) -> np.ndarray:
    """Symmetric diffusion on the connection-adjacency graph; conserves the total."""
    heads, tails = network.pairs[:, 0], network.pairs[:, 1]
    degree = network.degree()
    sums = _joint_sums(network, energy)
    neighbour_sum = sums[heads] + sums[tails] - 2 * energy
    neighbour_count = degree[heads] + degree[tails] - 2
    largest = max(int(neighbour_count.max(initial=0)), 1)
    return energy + rate / largest * (neighbour_sum - neighbour_count * energy)
