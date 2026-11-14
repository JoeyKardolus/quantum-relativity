"""The live loop: steps the simulation while started and keeps every open page in step.

Contract: `Runner(simulation)` serves page connections (`serve(socket)`) and
runs the stepping loop (`run()`, one asyncio task for the app's life). A new
page gets settings and shapes at once. Client messages: "start" (step at
`speed` steps per second), "pause", "reset" (seed, cells clamped to
CELL_RANGE; settings and shapes sent), "settings" (values applied, settings
echoed). While running, a shapes message (`shapes_message`, straight from the
engine) goes out at most SHAPES_PER_SECOND, and always on the step that pauses
the run. Every page gets every message (one shared simulation). Steps, resets
and shapes run in a worker thread under one lock, and messages are encoded
there too, so the server keeps answering. A bad message or a failed step is
printed on the console; a failed step pauses the run, and so does a step that
leaves no joint (every energy volume fell into one). A page that disconnects or
cannot be sent to is dropped.
"""
from __future__ import annotations

import asyncio
import json
import time
import traceback

from fastapi import WebSocket, WebSocketDisconnect

from physics.app.settings_message import settings_message
from physics.app.shapes_message import shapes_message
from physics.app.simulation import Simulation
from physics.cells import engine_stats

SHAPES_PER_SECOND = 8
CELL_RANGE = (50, 20000)


class Runner:
    """One simulation, the open pages, and the loop that steps it."""

    def __init__(self, simulation: Simulation) -> None:
        self.simulation = simulation
        self.started = False
        self._sockets: set[WebSocket] = set()
        self._lock = asyncio.Lock()
        self._wake = asyncio.Event()
        self._last_shapes = 0.0

    async def serve(self, socket: WebSocket) -> None:
        """Hold one page's connection until it closes."""
        await socket.accept()
        self._sockets.add(socket)
        try:
            async with self._lock:
                greeting = await asyncio.to_thread(self._encoded, settings_message(self.simulation),
                                                   shapes_message)
            for text in greeting:
                await socket.send_text(text)
            while True:
                await self._handle(await socket.receive_json())
        except WebSocketDisconnect:
            pass
        finally:
            self._sockets.discard(socket)

    async def run(self) -> None:
        """Step while started, at the requested speed, sending shapes when due."""
        while True:
            if not self.started:
                self._wake.clear()
                await self._wake.wait()
                continue
            begun = time.monotonic()
            for text in await self._advance(begun):
                await self._broadcast(text)
            await asyncio.sleep(max(0.0, 1.0 / self.simulation.speed - (time.monotonic() - begun)))

    async def _advance(self, begun: float) -> list[str]:
        async with self._lock:
            try:
                await asyncio.to_thread(self.simulation.advance)
            except Exception:
                traceback.print_exc()
                print("paused: the step failed", flush=True)
                self.started = False
                return []
            joints = await asyncio.to_thread(lambda: engine_stats(self.simulation.engine)["joints"])
            if joints == 0:
                print("paused: every energy volume fell into one, no joint is left", flush=True)
                self.started = False
            if self.started and begun - self._last_shapes < 1.0 / SHAPES_PER_SECOND:
                return []
            self._last_shapes = begun
            return await asyncio.to_thread(self._encoded, shapes_message)

    async def _handle(self, message: dict) -> None:
        kind = message.get("type") if isinstance(message, dict) else None
        try:
            if kind == "start":
                self.started = True
                self._wake.set()
            elif kind == "pause":
                self.started = False
            elif kind == "reset":
                await self._reset(int(message["seed"]), int(message["cells"]))
            elif kind == "settings":
                await self._settings(dict(message["values"]))
            else:
                raise ValueError(f"unknown message type {kind!r}")
        except (KeyError, TypeError, ValueError) as error:
            print(f"ignored message {message!r}: {error}", flush=True)

    async def _reset(self, seed: int, cells: int) -> None:
        cells = min(max(cells, CELL_RANGE[0]), CELL_RANGE[1])
        async with self._lock:
            await asyncio.to_thread(self.simulation.reset, seed, cells)
            outgoing = await asyncio.to_thread(self._encoded, settings_message(self.simulation),
                                               shapes_message)
        for text in outgoing:
            await self._broadcast(text)

    async def _settings(self, values: dict) -> None:
        async with self._lock:
            self.simulation.apply(values)
            echo = json.dumps(settings_message(self.simulation))
        await self._broadcast(echo)

    def _encoded(self, *messages) -> list[str]:
        """Each message as JSON text; a callable is first called with the simulation."""
        return [json.dumps(message(self.simulation) if callable(message) else message)
                for message in messages]

    async def _broadcast(self, text: str) -> None:
        for socket in list(self._sockets):
            try:
                await socket.send_text(text)
            except Exception:
                self._sockets.discard(socket)
