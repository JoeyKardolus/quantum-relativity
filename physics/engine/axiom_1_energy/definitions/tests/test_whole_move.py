"""A flow becomes whole quanta, the odd one by the phase's part of its step, never taking a box below 0."""
import numpy as np
import warp as wp

from physics.engine.axiom_1_energy.definitions.whole_move import odd_part, whole_move


@wp.kernel
def _moves(flow: wp.array(dtype=wp.float64), part: wp.array(dtype=wp.float64), out: wp.array(dtype=wp.int32)):
    out[wp.tid()] = whole_move(flow[wp.tid()], part[wp.tid()], 5, 4)   # the giver has 5, the taker 4


@wp.kernel
def _parts(phase: wp.array(dtype=wp.float32), steps: wp.int32, out: wp.array(dtype=wp.float64)):
    out[wp.tid()] = odd_part(phase[wp.tid()], steps)


def test_whole_moves_round_by_the_part_and_stop_at_an_empty_box():
    flows, parts = [2.3, 2.3, -2.3, 9.0, -9.0, 0.0], [0.6, 0.8, 0.2, 0.5, 0.5, 0.999]
    out = wp.zeros(6, dtype=wp.int32, device="cpu")
    wp.launch(_moves, dim=6, device="cpu", inputs=[wp.array(flows, dtype=wp.float64, device="cpu"),
                                                   wp.array(parts, dtype=wp.float64, device="cpu"), out])
    assert out.numpy().tolist() == [2, 3, -3, 5, -4, 0]


def test_the_odd_part_is_how_far_the_phase_stands_through_a_whole_step_of_its_turn():
    turn = 2 * np.pi
    phases = wp.array([0.0, turn / 4, -turn / 4, turn / 12, turn / 8], dtype=wp.float32, device="cpu")
    whole, sixths = (wp.zeros(5, dtype=wp.float64, device="cpu") for _ in range(2))
    wp.launch(_parts, dim=5, device="cpu", inputs=[phases, 1, whole])
    wp.launch(_parts, dim=5, device="cpu", inputs=[phases, 6, sixths])
    assert np.allclose(whole.numpy(), [0, 0.25, 0.75, 1 / 12, 0.125], atol=1e-6)
    assert np.allclose(sixths.numpy(), [0, 0.5, 0.5, 0.5, 0.75], atol=1e-5)  # six whole steps a turn
    assert whole.numpy().min() >= 0 and sixths.numpy().max() < 1
