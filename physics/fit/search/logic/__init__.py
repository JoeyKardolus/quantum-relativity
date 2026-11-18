"""The search, logic: what happens in a search.

  run_universe.py      run_universe: one wrapping universe stepped, measured and scored
  work_task.py         work_task: in a worker process, one task run and its record sent back
  load_records.py      load_records: every record written to a results file so far
  write_record.py      write_record: one run's record appended to the results file
  run_tasks.py         run_tasks: the tasks not yet done, a process per universe, side by side
  find_holders.py      find_holders: which searched points still stand when run four times longer
  run_search.py        run_search: the phases in order (defaults, random, refine, hold, sweep), resumable
  summarize_search.py  summarize_search: a plain-text summary of a search's records
"""
