"""Draw uniform: the engine's one source of random numbers, a hash of what is being drawn for.

Contract: `draw_uniform(seed, step, pass_id, index)` (Warp func; uint32, uint32, uint32,
uint64) returns a float64 in [0, 1). It keeps no state: the same four numbers always
give the same draw, so the same seed gives the same run on any device and in any thread
order. `pass_id` is from passes.py. `index` says which thing draws:
  a box               cell * 216 + box, box = x * 36 + y * 6 + z
  a pair of boxes     the index of its lower box
  a cell              its id
  a way               its slot
  a facing pair of boxes of a way  slot * 16 + its place in the mouth
  a half of a split     its volume's id * 2 ** 32 + the half's cell
The hash: start from `seed`, then mix in `step`, `pass_id` and `index` one after the
other, each time by xor followed by the splitmix64 finish; the top 53 bits over 2 ** 53.
  a split               its volume's id * 2 ** 32 + the number of the draw
`draw_uniform_host` is the host twin, bit for bit the same, and `draw_uniform_rows` the
same for a whole numpy array of indexes at once.
"""
from __future__ import annotations

import numpy as np
import warp as wp

wp.set_module_options({"fuse_fp": False})

_GOLD = 0x9E3779B97F4A7C15
_FIRST = 0xBF58476D1CE4E5B9
_SECOND = 0x94D049BB133111EB
_WORD = 2 ** 64
_GOLD_W = wp.constant(wp.uint64(_GOLD))
_FIRST_W = wp.constant(wp.uint64(_FIRST))
_SECOND_W = wp.constant(wp.uint64(_SECOND))
_UNIT = wp.constant(wp.float64(2.0 ** -53))


@wp.func
def _finish(value: wp.uint64) -> wp.uint64:
    """The splitmix64 finish: every bit of the input changes every bit of the output."""
    mixed = (value ^ (value >> wp.uint64(30))) * _FIRST_W
    mixed = (mixed ^ (mixed >> wp.uint64(27))) * _SECOND_W
    return mixed ^ (mixed >> wp.uint64(31))


@wp.func
def draw_uniform(seed: wp.uint32, step: wp.uint32, pass_id: wp.uint32, index: wp.uint64) -> wp.float64:
    """The draw in [0, 1) for one use (`pass_id`) by one thing (`index`) in one step of one seed."""
    mixed = _finish(wp.uint64(seed) + _GOLD_W)
    mixed = _finish(mixed ^ wp.uint64(step))
    mixed = _finish(mixed ^ wp.uint64(pass_id))
    mixed = _finish(mixed ^ index)
    return wp.float64(mixed >> wp.uint64(11)) * _UNIT


def _finish_host(value: int) -> int:
    """The splitmix64 finish (host)."""
    mixed = ((value ^ (value >> 30)) * _FIRST) % _WORD
    mixed = ((mixed ^ (mixed >> 27)) * _SECOND) % _WORD
    return mixed ^ (mixed >> 31)


def draw_uniform_host(seed: int, step: int, pass_id: int, index: int) -> float:
    """The same draw on the host."""
    mixed = _finish_host(_before_index(seed, step, pass_id) ^ int(index))
    return (mixed >> 11) * 2.0 ** -53


def _finish_rows(value: np.ndarray) -> np.ndarray:
    """The splitmix64 finish for a uint64 array (numpy's uint64 products wrap as the word does)."""
    mixed = (value ^ (value >> np.uint64(30))) * np.uint64(_FIRST)
    mixed = (mixed ^ (mixed >> np.uint64(27))) * np.uint64(_SECOND)
    return mixed ^ (mixed >> np.uint64(31))


def draw_uniform_rows(seed: int, step: int, pass_id: int, indexes) -> np.ndarray:
    """The same draw for every index of an array, as float64."""
    start = _before_index(seed, step, pass_id)
    mixed = _finish_rows(np.uint64(start) ^ np.asarray(indexes).astype(np.uint64))
    return (mixed >> np.uint64(11)).astype(np.float64) * 2.0 ** -53


def _before_index(seed: int, step: int, pass_id: int) -> int:
    """The hash after seed, step and pass are mixed in, before the index is."""
    mixed = _finish_host((int(seed) + _GOLD) % _WORD)
    mixed = _finish_host(mixed ^ int(step))
    return _finish_host(mixed ^ int(pass_id))
