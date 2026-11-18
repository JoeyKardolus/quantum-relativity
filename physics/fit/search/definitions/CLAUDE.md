# The search, definitions

What is searched and the marks a run and a point are judged by. Each file's docstring
is its contract.

- `ranges.py`: the ten searched settings and their ranges.
- `unit_to_overrides.py`: a point of the unit cube as engine settings.
- `latin_hypercube.py`, `refine_width.py`: where the random and the refine points lie.
- `search_plan.py`: the sizes of a search, one frozen record.
- `make_task.py`: one run to do, with its id.
- `torn_out.py`, `why_ended.py`: the marks at which a universe is over.
- `searched_phase.py`: which phases' runs are searched points.
- `point_holds.py`: when a point still stands in a longer run.

## Must not get wrong

- **Every default lies inside its range**, and both ends of every range are settings
  the engine accepts. A test checks both.
- **Every engine setting is either searched or named as not searched** in `ranges.py`.
  A new setting in the engine fails a test until it is put in one list.
- **No off switch.** The engine refuses 0 for a strength, so every log range starts above 0.
- **A task's id is its identity**: change how ids are made and a stopped search no
  longer resumes.
- **The marks are written down, not fitted.**
