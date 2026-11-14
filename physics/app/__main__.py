"""Command line: `python -m physics.app [--port 8765] [--joints 2500] [--seed 1]`.

Contract: builds the app (`create_app`) and serves it with uvicorn on
127.0.0.1 at the port, after printing the URL to open. WSL2 forwards
localhost, so a browser on Windows reaches the same URL.
"""
from __future__ import annotations

import argparse

import uvicorn

from physics.app.server import create_app


def main() -> None:
    """Parse the options, print the URL and serve until stopped."""
    parser = argparse.ArgumentParser(prog="python -m physics.app")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--joints", type=int, default=2500)
    parser.add_argument("--seed", type=int, default=1)
    arguments = parser.parse_args()
    print(f"quantum-relativity live: http://localhost:{arguments.port}", flush=True)
    uvicorn.run(create_app(arguments.joints, arguments.seed), host="127.0.0.1",
                port=arguments.port, log_level="warning")


if __name__ == "__main__":
    main()
