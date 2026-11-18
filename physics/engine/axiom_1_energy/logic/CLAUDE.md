# Axiom 1, logic

`flow_inside.py`: one step of the fluid in every cell that holds its own boxes. Six launches, three axes times two parities.

- No box may be in two pairs of one launch. That is why there is no second buffer.
- The launch runs over slots `0 .. count - 1` only. Calm cells are not visited.
- Nothing here reads an array back to the host.
