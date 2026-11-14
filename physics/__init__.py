"""Connection theory simulation (bible: physics/connection-theory). Tier M.

Module map:
  network/      joints and connections with energy and length; no coordinates
  measure/      dimension, distance, proper time, expansion, read from the network only
  rules/        energy-length, drift, attraction, joint repulsion; steps the network
  experiments/  one per milestone; each returns a result dict with measured vs known
  report/       build_page: one self-contained HTML results page, plots embedded
  __main__.py   python -m physics [--seed N] [--out DIR]: run all, write DIR/index.html
"""
