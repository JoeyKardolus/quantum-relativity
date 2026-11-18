# Axiom 3, logic

`take_own.py`, `give_back.py`, `boxes_of.py` (a cell's boxes whichever way it is kept) and `lay_square.py` (four volumes, four joints).

- Every second row of volumes is laid half a turn on (`_lay`): facing fluid then agrees along the first direction too.

- Dealing must sum to the cell's total to the quantum. The left-over quanta go one each from box 0 up.
- `give_back` tests the cell at the moment it is asked.
- `take_own` and `give_back` work one cell at a time from the host: a layer 1 stand-in. Layer 2 does both for many cells on the card; only the `_lay` kernel carries over.
- A joint (minus total) is dealt like a volume, the sign kept.
