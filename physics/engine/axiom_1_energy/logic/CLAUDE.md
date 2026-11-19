# Axiom 1, logic

`flow_inside.py`: one step of the fluid in every cell that holds its own boxes. Seven launches: two from layer to layer, three round the core, two along the ring.

- No box may be in two pairs of one launch. That is why there is no second buffer.
- Rows round the core and along the ring close on themselves. Ring to ring: only in the jet.
- The launch runs over slots `0 .. count - 1` only. Calm cells are not visited.
- Nothing here reads an array back to the host.
