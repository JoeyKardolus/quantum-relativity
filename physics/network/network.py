"""The Network: joints and connections that hold energy. No coordinates, ever.

Contract: a frozen value. Joints are the integers 0..joint_count-1 and carry
nothing but a charge. Connection i joins joints pairs[i, 0] and pairs[i, 1]
(never a joint to itself, never twice) and holds energy[i] and length[i]; the
energy-length rule keeps energy * length = LENGTH_SCALE unless a rule run with
that rule switched off has let them part. `causal` is set only for a spacetime
network: a sparse boolean matrix with causal[i, j] true when joint i precedes
joint j (the full transitive relation, not only the links). Rules return a new
Network via `dataclasses.replace`; nothing mutates one in place.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

from physics.network.energy_length import energy_length


@dataclass(frozen=True)
class Network:
    """Joints, connections with energy and length, joint charges, optional causal order."""

    joint_count: int
    pairs: np.ndarray
    energy: np.ndarray
    length: np.ndarray
    charge: np.ndarray = field(default=None)
    causal: sp.csr_matrix | None = None

    def __post_init__(self) -> None:
        if self.charge is None:
            object.__setattr__(self, "charge", np.zeros(self.joint_count))
        if self.pairs.ndim != 2 or self.pairs.shape[1] != 2:
            raise ValueError("pairs must have shape (connections, 2)")
        if not (len(self.energy) == len(self.length) == len(self.pairs)):
            raise ValueError("energy, length and pairs disagree on the connection count")
        if len(self.charge) != self.joint_count:
            raise ValueError("charge needs one value per joint")
        if np.any(self.pairs[:, 0] == self.pairs[:, 1]):
            raise ValueError("a connection cannot join a joint to itself")

    @classmethod
    def from_lengths(cls, joint_count: int, pairs: np.ndarray, length: np.ndarray,
                     **extra) -> "Network":
        """Network whose energies follow the given lengths by the energy-length rule."""
        length = np.asarray(length, dtype=float)
        return cls(joint_count, np.asarray(pairs, dtype=np.int64), energy_length(length),
                   length, **extra)

    @property
    def connection_count(self) -> int:
        return len(self.pairs)

    def adjacency(self, weight: np.ndarray | None = None) -> sp.csr_matrix:
        """Symmetric joint-by-joint matrix; entry = weight of the connection (default length)."""
        values = self.length if weight is None else weight
        heads, tails = self.pairs[:, 0], self.pairs[:, 1]
        matrix = sp.coo_matrix(
            (np.concatenate([values, values]),
             (np.concatenate([heads, tails]), np.concatenate([tails, heads]))),
            shape=(self.joint_count, self.joint_count))
        return matrix.tocsr()

    def degree(self) -> np.ndarray:
        """Number of connections at each joint."""
        return np.bincount(self.pairs.ravel(), minlength=self.joint_count)
