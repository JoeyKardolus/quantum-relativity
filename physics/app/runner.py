"""The live loop: steps the simulation while started and keeps every open page in step.

Contract: `Runner(simulation)` serves page connections (`serve(socket)`) and
runs the stepping loop (`run()`, one asyncio task for the app's life). A new
page gets, in order: settings, mesh, frame, stats. Client messages (JSON text):
"start" (step at `speed` steps per second), "pause", "reset" (seed, cells clamped
to CELL_RANGE; settings, mesh, frame and stats sent), "settings" (values applied,
settings echoed). Start and pause answer every page with a stats message, so all
pages show the same run state.
While running, an update goes out at most FRAMES_PER_SECOND, and always on the step
that pauses the run: a "mesh" message (JSON, page_mesh) first only when the topology
changed (mesh_key: joint collapse or volume split), then the binary frame (page_frame,
corner positions and pull and push bytes), then a stats message. Each mesh gets a new
mesh_id, carried by every frame drawn on it.
Every page gets every message (one shared simulation). Steps, resets and encoding run
in a worker thread under one lock, so the server keeps answering. A bad message or a
failed step is printed on the console; a failed step pauses the run, and so does a
step that leaves no joint (every energy volume fell into one). A page that
disconnects or cannot be sent to is dropped.
"""
from __future__ import annotations

import asyncio
import json
import time
import traceback

from fastapi import WebSocket, WebSocketDisconnect

from physics.app.mesh_key import mesh_key
from physics.app.page_frame import page_frame
from physics.app.page_mesh import page_mesh
from physics.app.settings_message import settings_message
from physics.app.simulation import Simulation
from physics.app.stats_message import stats_message
from physics.cells import engine_stats

FRAMES_PER_SECOND = 10
CELL_RANGE = (50, 20000)


class Runner:
    """One simulation, the open pages, and the loop that steps it."""

    def __init__(self, simulation: Simulation) -> None:
        self.simulation = simulation
        self.started = False
        self._sockets: set[WebSocket] = set()
        self._lock = asyncio.Lock()
        self._wake = asyncio.Event()
        self._last_frame = 0.0
        self._mesh_id = 0
        self._mesh_key: bytes | None = None

    async def serve(self, socket: WebSocket) -> None:
        """Hold one page's connection until it closes."""
        await socket.accept()
        self._sockets.add(socket)
        try:
            async with self._lock:
                greeting = await asyncio.to_thread(self._full_update)
            for message in greeting:
                await _send(socket, message)
            while True:
                await self._handle(await socket.receive_json())
        except WebSocketDisconnect:
            pass
        finally:
            self._sockets.discard(socket)

    async def run(self) -> None:
        """Step while started, at the requested speed, sending an update when due."""
        while True:
            if not self.started:
                self._wake.clear()
                await self._wake.wait()
                continue
            begun = time.monotonic()
            for message in await self._advance(begun):
                await self._broadcast(message)
            await asyncio.sleep(max(0.0, 1.0 / self.simulation.speed - (time.monotonic() - begun)))

    async def _advance(self, begun: float) -> list[str | bytes]:
        async with self._lock:
            try:
                await asyncio.to_thread(self.simulation.advance)
            except Exception:
                traceback.print_exc()
                print("paused: the step failed", flush=True)
                self.started = False
                return [json.dumps(stats_message(self.simulation, False))]
            joints = await asyncio.to_thread(lambda: engine_stats(self.simulation.engine)["joints"])
            if joints == 0:
                print("paused: every energy volume fell into one, no joint is left", flush=True)
                self.started = False
            if self.started and begun - self._last_frame < 1.0 / FRAMES_PER_SECOND:
                return []
            self._last_frame = begun
            return await asyncio.to_thread(self._update)

    async def _handle(self, message: dict) -> None:
        kind = message.get("type") if isinstance(message, dict) else None
        try:
            if kind == "start":
                self.started = True
                self._wake.set()
                await self._broadcast(json.dumps(stats_message(self.simulation, True)))
            elif kind == "pause":
                self.started = False
                await self._broadcast(json.dumps(stats_message(self.simulation, False)))
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
            self._mesh_key = None
            outgoing = await asyncio.to_thread(self._full_update)
        for message in outgoing:
            await self._broadcast(message)

    async def _settings(self, values: dict) -> None:
        async with self._lock:
            self.simulation.apply(values)
            echo = json.dumps(settings_message(self.simulation))
        await self._broadcast(echo)

    def _full_update(self) -> list[str | bytes]:
        """Settings, then the mesh (always), the frame and the stats."""
        return [json.dumps(settings_message(self.simulation)), self._mesh_text(),
                page_frame(self.simulation, self._mesh_id),
                json.dumps(stats_message(self.simulation, self.started))]

    def _update(self) -> list[str | bytes]:
        """The mesh only if its topology changed, then the frame and the stats."""
        outgoing: list[str | bytes] = []
        if mesh_key(self.simulation.engine) != self._mesh_key:
            outgoing.append(self._mesh_text())
        outgoing.append(page_frame(self.simulation, self._mesh_id))
        outgoing.append(json.dumps(stats_message(self.simulation, self.started)))
        return outgoing

    def _mesh_text(self) -> str:
        """A mesh message as compact JSON; a new mesh_id when the topology is new."""
        key = mesh_key(self.simulation.engine)
        if key != self._mesh_key:
            self._mesh_id += 1
            self._mesh_key = key
        return json.dumps(page_mesh(self.simulation, self._mesh_id), separators=(",", ":"))

    async def _broadcast(self, message: str | bytes) -> None:
        for socket in list(self._sockets):
            try:
                await _send(socket, message)
            except Exception:
                self._sockets.discard(socket)


async def _send(socket: WebSocket, message: str | bytes) -> None:
    """Text as a text message, bytes as a binary one."""
    if isinstance(message, bytes):
        await socket.send_bytes(message)
    else:
        await socket.send_text(message)
