# Axiom 3, logic

`take_own.py`, `give_back.py`, `boxes_of.py` (a cell's boxes whichever way it is kept) and `lay_grid.py` (the grid that wraps; `take_all` deals every cell at once).

- Along the direction after the swirl's, every second volume is laid half a turn on (`_lay`).
- Mirrors close only round an even number of volumes across the swirl.
- Dealing must sum to the cell's total to the quantum. The left-over quanta go one each from box 0 up.
- `give_back` tests the cell at the moment it is asked.
- `take_own` and `give_back` work one cell at a time from the host.
- A joint (minus total) is dealt like a volume, the sign kept.
