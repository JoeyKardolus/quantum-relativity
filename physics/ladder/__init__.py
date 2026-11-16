"""Detail ladder: blocks of 8^3 Planck boxes held at the detail their quietness allows, totals exact.

Bible `physics/quantum-relativity` rule 4 (each box one signed whole number: plus pulls,
minus pushes, 0 no space) and issue #3 (the detail ladder). A library the core engine
calls; it holds the quanta (int32) and the energy fluid's phase (float32) per box. Levels:
FULL every box; COARSE a quiet block as shared scales and 4-bit offsets per group of 4^3
(GSQ-style); REUSED a quiet block's exact content shared with every identical block
(Hashlife-style); EMPTY no space, no storage. Switching never changes a box's sign or the
plus and minus totals; blocks with a whirl or flow stay FULL.

Module map:
  settings.py           LadderSettings: quietness and error bounds; BLOCK, GROUP, BOXES
  level.py              Level: EMPTY, FULL, COARSE, REUSED
  store.py              LadderStore: block table and slot pools (why not NanoVDB)
  new_store.py          new_store: an all-EMPTY grid of blocks
  load_region.py        load_region: a dense field in as FULL blocks
  read_region.py        read_region: a dense field out, any level decoded
  gather_blocks.py      gather_blocks: blocks plus halo for the core's step, on the device
  write_blocks.py       write_blocks: the core's new block content, FULL where changed
  promote_blocks.py     promote_blocks: force blocks FULL now
  measure_quietness.py  measure_quietness: flow, whirls, space per block
  switch_levels.py      switch_levels: the once-a-step pass applying the rules
  total_quanta.py       total_quanta: exact plus and minus totals
  memory_per_box.py     memory_per_box: bytes per box at each level
  step_memo.py          StepMemo: the Hashlife table of one step rule
  memo_step.py          memo_step: a step rule run once per distinct block-with-halo
  _kernels.py, _moves.py, _pool.py: Warp kernels, level moves, slot pools
"""
from physics.ladder.settings import LadderSettings
from physics.ladder.level import Level
from physics.ladder.store import LadderStore
from physics.ladder.new_store import new_store
from physics.ladder.load_region import load_region
from physics.ladder.read_region import read_region
from physics.ladder.gather_blocks import gather_blocks
from physics.ladder.write_blocks import write_blocks
from physics.ladder.promote_blocks import promote_blocks
from physics.ladder.measure_quietness import measure_quietness
from physics.ladder.switch_levels import switch_levels
from physics.ladder.total_quanta import total_quanta
from physics.ladder.memory_per_box import memory_per_box
from physics.ladder.step_memo import StepMemo
from physics.ladder.memo_step import memo_step

__all__ = ["LadderSettings", "Level", "LadderStore", "new_store", "load_region", "read_region", "gather_blocks",
           "write_blocks", "promote_blocks", "measure_quietness", "switch_levels", "total_quanta",
           "memory_per_box", "StepMemo", "memo_step"]
