# Axiom 3, logic

`take_own.py`, `give_back.py`, and `boxes_of.py` (a cell's boxes whichever way it is kept).

- Dealing must sum to the cell's total to the quantum. The left-over quanta go one each from box 0 up.
- `give_back` tests the cell at the moment it is asked. Watching it for a whole round first comes with layer 2, when cells go back by themselves.
- `take_own` and `give_back` work one cell at a time from the host: a layer 1 stand-in. Layer 2 does both for many cells on the card; only the `_lay` kernel carries over.
- A joint (minus total) is dealt like a volume, the sign kept.
