"""GPU driver: point Mesa at the NVIDIA card through d3d12 before any OpenGL loads.

Contract: `use_gpu_driver()` sets GALLIUM_DRIVER=d3d12 and
MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA in os.environ unless already set (so a user
can override either), and returns the two values in force. Under WSL2 without them
Mesa draws on the CPU (llvmpipe). Must run before pyglet is imported.
"""
from __future__ import annotations

import os


def use_gpu_driver() -> dict[str, str]:
    """Set the two Mesa variables unless set, and return them."""
    os.environ.setdefault("GALLIUM_DRIVER", "d3d12")
    os.environ.setdefault("MESA_D3D12_DEFAULT_ADAPTER_NAME", "NVIDIA")
    return {name: os.environ[name] for name in ("GALLIUM_DRIVER", "MESA_D3D12_DEFAULT_ADAPTER_NAME")}
