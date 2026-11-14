"""The live loop: steps the simulation while started and keeps every open page in step.

Contract: `Runner(simulation)` serves page connections (`serve(socket)`) and
runs the stepping loop (`run()`, one asyncio task for the app's life). A new
page gets settings, mesh and frame at once. Client messages: "start" (step
at `speed` steps per second, mesh sent), "pause", "reset" (seed, joints
clamped to JOINT_RANGE; settings, mesh and frame sent), "settings" (values
applied, settings echoed). Every page gets every message (one shared
simulation). Frames go out at most FRAMES_PER_SECOND; a mesh goes out after
any step that changed the connection list. Steps, resets and frames run in a
worker thread under one lock, so the server keeps answering. A bad message or
a failed step is printed on the console; a failed step pauses the run, and so does a
step that leaves no connection (every joint fell into one). A page
that disconnects or cannot be sent to is dropped.
"""
from __future__ import annotations

import asyncio
import json
import time
import traceback

from fastapi import WebSocket, WebSocketDisconnect

from physics.app.frame_message import frame_message
from physics.app.mesh_message import mesh_message
from physics.app.settings_message import settings_message
from physics.app.simulation import Simulation

FRAMES_PER_SECOND = 15
JOINT_RANGE = (50, 20000)


class Runner:
    """One simulation, the open pages, and the loop that steps it."""

    def __init__(self, simulation: Simulation) -> None:
        self.simulation = simulation
        self.started = False
        self._sockets: set[WebSocket] = set()
        self._lock = asyncio.Lock()
        self._wake = asyncio.Event()
        self._last_frame = 0.0

    async def serve(self, socket: WebSocket) -> None:
        """Hold one page's connection until it closes."""
        await socket.accept()
        self._sockets.add(socket)
        try:
            async with self._lock:
                frame = await asyncio.to_thread(frame_message, self.simulation)
                greeting = [settings_message(self.simulation), mesh_message(self.simulation), frame]
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
            if self.simulation.network.connection_count == 0:
                print("paused: every joint fell into one, no connection is left", flush=True)
                self.started = False
            if not self.started or begun - self._last_frame >= 1.0 / FRAMES_PER_SECOND:
                self._last_frame = begun
                outgoing.append(await asyncio.to_thread(frame_message, self.simulation))
        return outgoing

    async def _handle(self, message: dict) -> None:
        kind = message.get("type") if isinstance(message, dict) else None
        try:
            if kind == "start":
                await self._start()
            elif kind == "pause":
                self.started = False
            elif kind == "reset":
                await self._reset(int(message["seed"]), int(message["joints"]))
            elif kind == "settings":
                await self._settings(dict(message["values"]))
            else:
                raise ValueError(f"unknown message type {kind!r}")
        except (KeyError, TypeError, ValueError) as error:
            print(f"ignored message {message!r}: {error}", flush=True)

    async def _start(self) -> None:
        self.started = True
        self._wake.set()
        await self._broadcast(mesh_message(self.simulation))

    async def _reset(self, seed: int, joints: int) -> None:
        joints = min(max(joints, JOINT_RANGE[0]), JOINT_RANGE[1])
        async with self._lock:
            await asyncio.to_thread(self.simulation.reset, seed, joints)
            frame = await asyncio.to_thread(frame_message, self.simulation)
            outgoing = [settings_message(self.simulation), mesh_message(self.simulation), frame]
        for message in outgoing:
            await self._broadcast(message)

    async def _settings(self, values: dict) -> None:
        async with self._lock:
            self.simulation.apply(values)
            echo = settings_message(self.simulation)
        await self._broadcast(echo)

    async def _broadcast(self, message: dict) -> None:
        text = json.dumps(message)
        for socket in list(self._sockets):
            try:
                await socket.send_text(text)
            except Exception:
                self._sockets.discard(socket)
