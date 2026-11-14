"""The live loop: steps the simulation while started and keeps every open page in step.

Contract: `Runner(simulation)` serves page connections (`serve(socket)`) and
runs the stepping loop (`run()`, one asyncio task for the app's life). A new
page gets settings, mesh and frame at once. Client messages: "start" (step
at `speed` steps per second, mesh sent), "pause", "reset" (seed, cells
clamped to CELL_RANGE; settings, mesh and frame sent), "settings" (values
applied, settings echoed), "slice" (depth clamped to 0..1, axis "x", "y" or "z";
the cut moves and a fresh slice is sent). Every frame is followed by its slice
(`slice_message` of the frame's own positions, radii and shades), and the frame's
stats gain "walls_not_matching_joints": the slice's walls between energy volumes
that are not neighbours in the grid. Every page gets every message (one shared
simulation). Frames go out at most FRAMES_PER_SECOND; a mesh goes out after
any step that changed the joint list. Steps, resets and frames run in a
worker thread under one lock, so the server keeps answering. A bad message or
a failed step is printed on the console; a failed step pauses the run, and so does a
step that leaves no joint (every energy volume fell into one). A page
that disconnects or cannot be sent to is dropped.
"""
from __future__ import annotations

import asyncio
import json
import math
import time
import traceback

import numpy as np
from fastapi import WebSocket, WebSocketDisconnect

from physics.app.frame_message import frame_message
from physics.app.mesh_message import mesh_message
from physics.app.settings_message import settings_message
from physics.app.slice_message import slice_message
from physics.app.simulation import Simulation

FRAMES_PER_SECOND = 15
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
        self.slice_depth = 0.5
        self.slice_axis = "z"
        self._shown: tuple[dict, object] | None = None

    async def serve(self, socket: WebSocket) -> None:
        """Hold one page's connection until it closes."""
        await socket.accept()
        self._sockets.add(socket)
        try:
            async with self._lock:
                frames = await asyncio.to_thread(self._frames)
                greeting = [settings_message(self.simulation), mesh_message(self.simulation), *frames]
            for message in greeting:
                await socket.send_text(json.dumps(message))
            while True:
                await self._handle(await socket.receive_json())
        except WebSocketDisconnect:
            pass
        finally:
            self._sockets.discard(socket)

    async def run(self) -> None:
        """Step while started, at the requested speed, sending meshes and frames."""
        while True:
            if not self.started:
                self._wake.clear()
                await self._wake.wait()
                continue
            begun = time.monotonic()
            outgoing = await self._advance(begun)
            for message in outgoing:
                await self._broadcast(message)
            await asyncio.sleep(max(0.0, 1.0 / self.simulation.speed - (time.monotonic() - begun)))

    async def _advance(self, begun: float) -> list[dict]:
        async with self._lock:
            try:
                changed = await asyncio.to_thread(self.simulation.advance)
            except Exception:
                traceback.print_exc()
                print("paused: the step failed", flush=True)
                self.started = False
                return []
            outgoing = [mesh_message(self.simulation)] if changed else []
            if len(self.simulation.grid.joints) == 0:
                print("paused: every energy volume fell into one, no joint is left", flush=True)
                self.started = False
            if not self.started or begun - self._last_frame >= 1.0 / FRAMES_PER_SECOND:
                self._last_frame = begun
                outgoing.extend(await asyncio.to_thread(self._frames))
        return outgoing

    async def _handle(self, message: dict) -> None:
        kind = message.get("type") if isinstance(message, dict) else None
        try:
            if kind == "start":
                await self._start()
            elif kind == "pause":
                self.started = False
            elif kind == "reset":
                await self._reset(int(message["seed"]), int(message["cells"]))
            elif kind == "settings":
                await self._settings(dict(message["values"]))
            elif kind == "slice":
                await self._move_slice(float(message["depth"]), str(message["axis"]))
            else:
                raise ValueError(f"unknown message type {kind!r}")
        except (KeyError, TypeError, ValueError) as error:
            print(f"ignored message {message!r}: {error}", flush=True)

    async def _start(self) -> None:
        self.started = True
        self._wake.set()
        await self._broadcast(mesh_message(self.simulation))

    async def _reset(self, seed: int, cells: int) -> None:
        cells = min(max(cells, CELL_RANGE[0]), CELL_RANGE[1])
        async with self._lock:
            await asyncio.to_thread(self.simulation.reset, seed, cells)
            frames = await asyncio.to_thread(self._frames)
            outgoing = [settings_message(self.simulation), mesh_message(self.simulation), *frames]
        for message in outgoing:
            await self._broadcast(message)

    async def _settings(self, values: dict) -> None:
        async with self._lock:
            self.simulation.apply(values)
            echo = settings_message(self.simulation)
        await self._broadcast(echo)

    async def _move_slice(self, depth: float, axis: str) -> None:
        if not math.isfinite(depth) or axis not in ("x", "y", "z"):
            raise ValueError(f"slice needs a finite depth and axis x, y or z, got {depth!r}, {axis!r}")
        async with self._lock:
            self.slice_depth, self.slice_axis = min(max(depth, 0.0), 1.0), axis
            if self._shown is None:
                return
            cut = await asyncio.to_thread(self._slice, *self._shown)
        await self._broadcast(cut)

    def _frames(self) -> list[dict]:
        """The frame and its slice, the slice's unmatched wall count added to the frame's stats."""
        frame = frame_message(self.simulation)
        self._shown = (frame, self.simulation.grid.joints.copy())
        cut = self._slice(*self._shown)
        frame["stats"]["walls_not_matching_joints"] = sum(wall["squeeze"] is None
                                                          for wall in cut["walls"])
        return [frame, cut]

    def _slice(self, frame: dict, joints) -> dict:
        return slice_message(np.asarray(frame["positions"], dtype=float).reshape(-1, 3),
                             frame["radius"], frame["energy"], joints, frame["squeeze"],
                             self.slice_depth, self.slice_axis, frame["step"])

    async def _broadcast(self, message: dict) -> None:
        text = json.dumps(message)
        for socket in list(self._sockets):
            try:
                await socket.send_text(text)
            except Exception:
                self._sockets.discard(socket)
