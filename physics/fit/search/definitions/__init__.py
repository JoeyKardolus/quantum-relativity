"""The search, definitions: what is searched and the marks a run and a point are judged by.

  ranges.py             LOG_RANGES, LINEAR_RANGES, SEARCHED: the settings searched and their ranges
  unit_to_overrides.py  unit_to_overrides: a point of the unit cube as engine settings
  latin_hypercube.py    latin_hypercube: the random phase's points, one per stratum on every axis
  refine_width.py       refine_width: how wide a refine round steps round its best points
  make_task.py          make_task: one run to do, with its id
  torn_out.py           torn_out: more than two tears per start volume
  why_ended.py          why_ended: the three marks at which a universe is over, in words
  searched_phase.py     searched_phase: which phases' runs are searched points
  search_plan.py        SearchPlan: the sizes of a search
  point_holds.py        point_holds: when a searched point still stands in a run four times longer
"""
