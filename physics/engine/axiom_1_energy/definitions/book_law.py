"""Book law: the total is not fixed, but every change of it is booked exactly.

Contract: `book_law(start, now)` (host) takes the books as read at the start and now
(dicts of whole numbers: `plus`, `minus` and `amount` are the plus quanta, the minus
quanta counted as a size, and both together; `now` also holds the counts `made_total`,
`taken_joint`, `taken_volumes`, `short_quanta`, `torn`) and returns `{"amount", "plus",
"minus", "taken"}`: what is there now less what the law says must be there. All four
are 0 while the books are exact.
  amount = start + made_total - taken_joint - taken_volumes
  plus   = start plus - taken_volumes - torn
  minus  = start minus + made_total - taken_joint + torn
  taken  : taken_volumes = taken_joint - short_quanta
A squeezed joint makes minus energy out of itself (made_total); a collapse takes the
joint's energy with it (taken_joint) and as much off the merged volume (taken_volumes);
a tear turns plus into minus (torn) and changes no amount. Nothing else changes a total.
`taken` ties the two takes together: a collapse takes off the rest exactly what its
joint held, less only what the merged volume did not have (`short_quanta`, booked
whole), so the two counts are not free of each other.
"""
from __future__ import annotations


def book_law(start: dict, now: dict) -> dict:
    """What the books hold now less what the law says, per total; all 0 when exact."""
    return {
        "amount": now["amount"] - (start["amount"] + now["made_total"] - now["taken_joint"] - now["taken_volumes"]),
        "plus": now["plus"] - (start["plus"] - now["taken_volumes"] - now["torn"]),
        "minus": now["minus"] - (start["minus"] + now["made_total"] - now["taken_joint"] + now["torn"]),
        "taken": now["taken_volumes"] - (now["taken_joint"] - now["short_quanta"]),
    }
