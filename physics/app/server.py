"""The FastAPI app: the page at GET / and the live connection at /ws.

Contract: `create_app(cells, seed)` builds the simulation and its Runner when
the app starts (lifespan), runs the stepping loop as a task until shutdown,
serves static/index.html at GET / (503 with a plain note while that file is
missing) and hands each /ws connection to the Runner.
"""
from __future__ import annotations

import asyncio
import contextlib
from pathlib import Path

from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse, PlainTextResponse, Response

from physics.app.runner import Runner
from physics.app.simulation import Simulation

PAGE = Path(__file__).parent / "static" / "index.html"


def create_app(cells: int = 2500, seed: int = 1) -> FastAPI:
    """The app serving the page and the simulation's WebSocket."""

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.runner = Runner(await asyncio.to_thread(Simulation, cells, seed))
        task = asyncio.create_task(app.state.runner.run())
        yield
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    app = FastAPI(lifespan=lifespan)

    @app.get("/")
    async def page() -> Response:
        if not PAGE.exists():
            return PlainTextResponse("The page (physics/app/static/index.html) is not there yet.",
                                     status_code=503)
        return FileResponse(PAGE)

    @app.websocket("/ws")
    async def live(socket: WebSocket) -> None:
        await app.state.runner.serve(socket)

    return app
