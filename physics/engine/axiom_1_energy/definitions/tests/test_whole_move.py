"""A flow becomes whole quanta, rounded by the draw and never taking a box below 0."""
import warp as wp

from physics.engine.axiom_1_energy.definitions.whole_move import whole_move


@wp.kernel
def _moves(flow: wp.array(dtype=wp.float64), uniform: wp.array(dtype=wp.float64), out: wp.array(dtype=wp.int32)):
    out[wp.tid()] = whole_move(flow[wp.tid()], uniform[wp.tid()], 5, 4)   # the giver has 5, the taker 4


def test_whole_moves_round_by_the_draw_and_stop_at_an_empty_box():
    flows, uniforms = [2.3, 2.3, -2.3, 9.0, -9.0, 0.0], [0.6, 0.8, 0.2, 0.5, 0.5, 0.999]
    out = wp.zeros(6, dtype=wp.int32, device="cpu")
    wp.launch(_moves, dim=6, device="cpu", inputs=[wp.array(flows, dtype=wp.float64, device="cpu"),
                                                   wp.array(uniforms, dtype=wp.float64, device="cpu"), out])
    assert out.numpy().tolist() == [2, 3, -3, 5, -4, 0]
