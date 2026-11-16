"""Step memo: the Hashlife table of one step rule, block-with-halo content to the block's next content.

Contract: `StepMemo(rule_key)` holds `table`, exact key bytes (the rule key, then a
gathered block's quanta and phase bytes with its halo) mapped to the step's result for
the block (quanta, phase host arrays of shape (8, 8, 8)); `hits` and `misses` count
lookups. `rule_key` names the step rule and its settings: a memo is valid for one rule
only, so a change of settings needs a new memo. When the table reaches `max_entries`
it is cleared before the next store (a plain bound, no eviction order).
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class StepMemo:
    """Exact-content cache of one step rule's per-block results."""

    rule_key: bytes
    max_entries: int = 1 << 16
    table: dict = field(default_factory=dict)
    hits: int = 0
    misses: int = 0
