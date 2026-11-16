"""Axiom 6, Limits: squeezed minus collapses into a merge of its two volumes; overstretched plus tears, and only that cell splits.

The owner's axiom (bible physics/quantum-relativity, "The axioms", 2026-11-16), checked in 5 parts, each
kept whole from the file it came from (names that two parts shared carry a _P<n> suffix).

Part 1 (was axiom 02). Plus and minus never cancel; only joint collapse turns minus into plus, only a tear turns plus into minus.

Bible rule 4. Default universe (seed 1, Settings(), axiom_run.py), 300 steps. After every
step: every box whose sign turned minus to plus carries the collapse mark, every box
whose sign turned plus to minus carries the split mark (a box that changed cell, emptied
and refilled by its new cell in one step, is counted apart and printed: its quanta did
not turn), and the plus total changed by
exactly the quanta collapse turned plus minus the quanta tears turned minus (plus and
minus never cancel: no other rule changes the plus total). Staged trigger: the 8-volume
even start of axiom 6 (new_universe(5, 10, seed=1, unevenness=0, margin=8)) with joint
J between volumes (2, 2, 2) and (4, 2, 2) packed with 30000 quanta per box, past
Settings().collapse_packing (27000; owner, 2026-11-16: a joint collapses only when its
energy concentration gets too high); one full step():
every box that turned minus to plus is marked as collapse, none turned plus to minus
and nothing tore, and the plus gain equals the collapsed quanta, the amount exact. A box
that changed cell in the same step (emptied and refilled by its new cell, owner-change
mark 2) is counted apart in both directions, as in the default run: its quanta did not
turn (until 2026-11-16 the plus-to-minus count forgot this, against this docstring; the
plus total, which only a turned quantum can change, is checked exactly).

Part 2 (was test_a_box_changes_cell_only_once_empty_and_never_holds_two_signs). A box changes cell only once its energy has flowed out, so no box ever holds two signs; only collapse turns minus plus.

Owner, 2026-11-16: the Planck boxes stay; when a cell's boxes follow its size a box
changes owner only after its energy has flowed out (exchange.py and mortar_shift.py
empty it into its own cell and refill it from the new one inside one move, so no box
ever holds two signs); plus and minus never cancel; only joint collapse turns minus
into plus and only a tear plus into minus. The start new_universe(5, 8, seed=1,
margin=8) runs 150 steps with the default settings. After every step: every box's sign
is its cell's kind (state.kind: plus in a volume, minus in a joint, 0 in empty space);
the amount ignoring sign is the start's; the plus quanta have grown by exactly the
minus quanta collapse turned plus less the plus quanta tears turned minus; and every box
that changed sign is collapse marked (bit 1), tear marked (bit 8) or changed cell (bit
2). Every 50 steps the joint boxes form one face-connected web and no plus box has an
empty box or a wall among its 26 neighbours (every volume stays wrapped). Boxes did
change cell.

Part 3 (was axiom 10). A joint is squeezed when smaller than its energy wants; push rises steeply; packed past the limit it collapses into ONE cell at once, and never for being low on energy.

Audit answers (owner, 2026-11-16): a joint is squeezed when it is smaller than its energy
wants; a joint collapses only when its energy concentration gets too high, never
because it is low on energy or smaller than its energy wants; when a joint collapses,
its two volumes become one cell at once. Bible rule 8:
push rises steeply, like a hard wall. Staged on new_universe(5, 10, seed=1,
unevenness=0, margin=8) (8 energy volumes, every box 1000, at rest), joint J at lattice
(3, 2, 2) between energy volumes (2, 2, 2) and (4, 2, 2), Settings():
  1. steep push: J's quanta scaled so its wanted size is 1, 2, 4 and 8 times its size
     (wanted side = size_energy / quanta, bible rule 4, the joints' constant
     state.size_energy[0, 1]); the engine's own cell push (axiom_pressure.py) rises with
     that ratio at least as its cube at every step up
     (steeper than a gas pressed at fixed temperature, which rises as the ratio)
  2. collapse: J packed with 30000 quanta per box, past collapse_packing (27000).
     Within 30 steps J collapses, and at the step it does the lattice holds 7 energy
     volumes (its two became one cell) and J holds no minus quanta
  3. the trigger: J left with 100 quanta per box (a tenth of the start's), so it wants a
     side 10 times longer than it has, a thousand times its room (squeezed far past the
     squeeze of 2 that collapsed a joint until 2026-11-16): collapse() takes nothing;
     packed with 30000 per box it takes J, one collapse

Part 4 (was axiom 11). A too-big volume tears; only that volume splits, its middle plus quanta turn minus into the new joint, ABAB holds after.

Audit answers (owner, 2026-11-16): a tear turns the middle of the torn volume's plus
quanta into minus to make the new joint, the mirror of collapse; only the too-big volume
splits, never its neighbours. Bible rule 8: the trigger is size. Default universe (seed
1, Settings(), axiom_run.py): the energy volumes alive after every step are the start's
plus the splits minus the collapses so far (each split adds exactly one volume; the
run is printed with its split count). Staged trigger, Settings() (split_across unchanged):
new_universe(5, 10, seed=1, unevenness=0, margin=10), 8 energy volumes, with energy
volume V (lattice 2, 2, 2) left with 100 quanta per box, so it wants a side 10 times
longer and grows, and energy leaves the volumes round it, so they want more room too. Up
to 300 steps until the first split; at that step:
  - the cell graph holds exactly one more energy volume (one volume split)
  - every box that got the split mark belonged before the step to one energy volume, the
    one with the most boxes, at least split_across ** 3 of them (only the too-big volume
    tears, its neighbours unchanged; since 2026-11-16 the first to tear here is volume
    (4, 2, 4), which loses energy toward V, not V itself)
  - the minus total rose by exactly the quanta the tear turned minus, and that is more
    than 0 (the tear's plus quanta became the new joint, amount kept)
  - ABAB: 0 touching box pairs of two volumes, 0 volume boxes beside empty space, the
    joints one web
  - the steps after (bug found 2026-11-16: the step after the split collapsed joint
    pieces and broke the web into 203 pieces, 28 plus boxes beside empty space): from
    the split on, 15 steps, the joints stay one web, no two volumes touch and no volume
    box is beside empty space, and no joint collapses (owner, 2026-11-16: a fresh joint
    from a tear holds the least energy it will ever have and must not collapse straight
    back; until then the tear merged back 3 times in 15 steps)

Part 5 (was test_the_default_start_has_no_collapse_in_its_first_20_steps). The default start holds no joint past the seal, and no joint collapses in its first 20 steps.

Bible foundation: the start is an actual grid of energy volumes wrapped in joints, the
unevenness in slightly different energies, and one correction lets sizes follow energy,
so no joint starts squeezed. The default start (new_universe(), seed 1) has every joint's
squeeze (wanted length over length, minus 1; stats()) below the seal, and 20 steps of
the default settings collapse no joint.
"""
import numpy as np

from physics.engine import (Settings, State, cell_count, cell_laws, cells, collapse, group_label, host_boxes, near,
                            new_universe, stats, step, totals)
from physics.engine.tests._gpu import needs_gpu
from physics.engine.tests.axiom_pressure import axiom_pressure
from physics.engine.tests.axiom_contacts import axiom_contacts


J_P1 = (3 * 7 + 2) * 7 + 2


@needs_gpu
def test_every_sign_change_in_the_default_run_is_a_collapse_or_a_tear():
    from physics.engine.tests.axiom_run import axiom_run
    run = axiom_run()
    previous_plus, previous_collapsed, previous_torn = run["start"]["plus"], 0, 0
    unmarked, unexplained, flips = 0, 0, np.zeros(6, dtype=np.int64)
    for row in run["steps"]:
        flips += row["flips"]
        unmarked += row["flips"][2] + row["flips"][5]
        expected = (row["tallies"][3] - previous_collapsed) - (row["torn"] - previous_torn)
        unexplained += abs((row["plus"] - previous_plus) - expected)
        previous_plus, previous_collapsed, previous_torn = row["plus"], row["tallies"][3], row["torn"]
    print(f"300 steps: boxes minus->plus by collapse {flips[0]}, by rehoming (emptied, refilled) {flips[1]}, "
          f"by nothing marked {flips[2]}; plus->minus by tear {flips[3]}, by rehoming {flips[4]}, by nothing "
          f"marked {flips[5]}; plus-total change not explained by collapse or tear: {unexplained} quanta")
    assert unmarked == 0 and unexplained == 0


@needs_gpu
def test_a_staged_collapse_is_the_only_minus_to_plus_and_keeps_the_amount():
    boxes = host_boxes(new_universe(5, 10, seed=1, unevenness=0.0, margin=8))
    quanta = boxes["quanta"].copy()
    quanta[boxes["owner"] == J_P1] = -30000
    graph = {name: boxes[name] for name in ("owner", "kind", "between", "place", "size", "size_energy")}
    state = State.from_host(quanta, np.zeros(quanta.shape), seed=1, lattice=graph)
    before = totals(state)
    step(state, Settings())
    after = totals(state)
    now, marks = state.quanta.numpy(), state.marks.numpy()
    turned_plus = (quanta < 0) & (now > 0)
    turned_minus = (quanta > 0) & (now < 0)
    rehomed_minus = int((turned_minus & ((marks & 2) == 2)).sum())
    collapsed = int(state.tallies.numpy()[0, 3])
    collapsed_boxes = int((turned_plus & ((marks & 1) == 1)).sum())
    unmarked = int((turned_plus & ((marks & 3) == 0)).sum())
    print(f"minus->plus {int(turned_plus.sum())} boxes: collapse-marked {collapsed_boxes}, changed cell "
          f"{int(turned_plus.sum()) - collapsed_boxes - unmarked}, neither {unmarked}; "
          f"plus->minus {int(turned_minus.sum())} (changed cell {rehomed_minus}); plus gain {after['plus'] - before['plus']}, collapsed quanta "
          f"{collapsed}; amount {before['amount']} -> {after['amount']}")
    assert collapsed_boxes > 0 and unmarked == 0
    assert turned_minus.sum() == rehomed_minus and int(state.torn[0]) == 0
    assert after["plus"] - before["plus"] == collapsed and after["amount"] == before["amount"]


@needs_gpu
def test_owner_changes_keep_every_box_on_its_cell_sign():
    state = new_universe(5, 8, seed=1, margin=8)
    settings = Settings()
    start = totals(state)
    before = np.sign(state.quanta.numpy())
    for count in range(150):
        step(state, settings)
        quanta = state.quanta.numpy()
        owner = state.owner.numpy()
        kinds = np.concatenate([state.kind[0].astype(np.int64), [0]])
        now = totals(state)
        turned = int(state.tallies.numpy()[0, 3]) - int(state.torn[0])
        after = np.sign(quanta)
        marks = state.marks.numpy()
        assert np.array_equal(after, kinds[owner])
        assert now["amount"] == start["amount"]
        assert now["plus"] - start["plus"] == turned and start["minus"] - now["minus"] == turned
        assert np.all(marks[before != after] & 11)
        before = after
        if count % 50 == 49:
            assert group_label(quanta < 0, state.wrap)[1] == 1
            assert not np.any((quanta > 0) & near(quanta == 0, state.wrap, wall=True))
            print(f"step {count + 1}: owner changes {int(state.tallies.numpy()[0, 2])}, turned plus {turned}")
    assert int(state.tallies.numpy()[0, 2]) > 0


J_P3 = (3 * 7 + 2) * 7 + 2


def _start(joint_packing: int) -> State:
    """The 8-volume even start with J_P3 at `joint_packing` minus quanta per box."""
    boxes = host_boxes(new_universe(5, 10, seed=1, unevenness=0.0, margin=8))
    quanta = boxes["quanta"].copy()
    quanta[boxes["owner"] == J_P3] = -joint_packing
    graph = {name: boxes[name] for name in ("owner", "kind", "between", "place", "size", "size_energy")}
    return State.from_host(quanta, np.zeros(quanta.shape, dtype=np.float32), seed=1, lattice=graph)


@needs_gpu
def test_a_joint_smaller_than_its_energy_wants_pushes_steeply():
    settings = Settings()
    ratios, pushes = [], []
    for wanted in (1, 2, 4, 8):
        state = _start(int(round(1000 * wanted ** (-1.0 / 3.0))))
        pressure, size, amount = axiom_pressure(state, settings)
        ratios.append((state.size_energy[0, 1] / amount[J_P3]) ** 3 / size[J_P3])
        pushes.append(pressure[J_P3])
    slopes = np.diff(np.log(pushes)) / np.diff(np.log(ratios))
    print("wanted/size " + " ".join(f"{value:.3f}" for value in ratios) + "; push "
          + " ".join(f"{value:.4g}" for value in pushes) + "; log slope " + " ".join(f"{value:.2f}" for value in slopes))
    assert np.all(slopes >= 3.0)


@needs_gpu
def test_a_joint_packed_past_the_limit_collapses_and_its_two_volumes_become_one_cell():
    settings = Settings()
    state = _start(30000)
    first, volumes_then, minus_left = None, None, None
    for count in range(1, 31):
        step(state, settings)
        if first is None and int(state.tallies.numpy()[0, 1]) > 0:
            first = count
            found = cells(host_boxes(state))
            volumes_then = found["volumes"]
            minus_left = int(-host_boxes(state)["quanta"][host_boxes(state)["owner"] == J_P3].clip(max=0).sum())
    measured = stats(state, settings)
    print(f"J_P3 at 30000 per box: first collapse at step {first}; energy volumes then {volumes_then}; J_P3's minus quanta "
          f"then {minus_left}; after 30 steps collapses {measured['collapses']}, energy volumes {measured['cells']}, "
          f"squeeze by packing max {measured['squeeze_max']:.3f}")
    assert first is not None
    assert volumes_then == 7 and minus_left == 0


@needs_gpu
def test_a_joint_low_on_energy_is_never_collapsed_only_one_packed_too_high():
    settings = Settings()
    taken = {}
    for packing in (100, 30000):
        state = _start(packing)
        cell_count(state)
        squeeze = cell_laws(state, settings)[0, J_P3]
        taken[packing] = (collapse(state, settings), float(squeeze), int(state.kind[0, J_P3]))
    print("J_P3 per box: (collapses, squeeze before, kind after) " + "; ".join(
        f"{packing}: {value}" for packing, value in taken.items()))
    assert taken[100][1] > 2.0
    assert taken[100][0] == 0 and taken[100][2] == -1
    assert taken[30000][0] == 1 and taken[30000][2] == 0


V = (2 * 7 + 2) * 7 + 2


@needs_gpu
def test_the_default_run_splits_nothing_but_the_too_big():
    from physics.engine.tests.axiom_run import axiom_run
    run = axiom_run()
    last = run["steps"][-1]
    start = int((run["kinds"] == 1).sum())
    wrong = [row["step"] for row in run["steps"] if row["volumes"] != start + row["splits"] - row["collapses"]]
    print(f"default run, 300 steps: splits {last['splits']}, torn {last['torn']}, collapses {last['collapses']}, "
          f"energy volumes {start} -> {last['volumes']}; steps where the count is off: {len(wrong)}")
    assert not wrong


@needs_gpu
def test_only_the_too_big_volume_splits_and_its_tear_becomes_the_joint():
    settings = Settings()
    boxes = host_boxes(new_universe(5, 10, seed=1, unevenness=0.0, margin=10))
    quanta = boxes["quanta"].copy()
    quanta[boxes["owner"] == V] = 100
    graph = {name: boxes[name] for name in ("owner", "kind", "between", "place", "size", "size_energy")}
    state = State.from_host(quanta, np.zeros(quanta.shape, dtype=np.float32), seed=1, lattice=graph)
    split_at, across = None, []
    for count in range(1, 301):
        before_owner = host_boxes(state)["owner"]
        cells_kind_before = state.kind[0].copy()
        before_volumes = int((state.kind[0] == 1).sum())
        before = totals(state)
        torn_before = int(state.torn[0])
        across.append(float(np.cbrt((host_boxes(state)["owner"] == V).sum())))
        step(state, settings)
        if state.splits[0] > 0:
            split_at = count
            break
    if split_at is None:
        print(f"no split in 300 steps; V's size across (boxes) {across[0]:.2f} .. {max(across):.2f}, at the end "
              f"{across[-1]:.2f}; split_across {settings.split_across}")
        assert split_at is not None
    after = totals(state)
    volumes_after = int((state.kind[0] == 1).sum())
    marks = state.marks.numpy()
    torn_cells = np.unique(before_owner[(marks & 8) == 8])
    torn_cell = int(torn_cells[0]) if torn_cells.size == 1 else -1
    held = np.bincount(before_owner[before_owner >= 0].ravel(), minlength=state.kind.shape[1])
    volume_kinds = cells_kind_before
    biggest = int(np.argmax(np.where(volume_kinds == 1, held[:volume_kinds.size], -1)))
    others_marked = int((((marks & 8) == 8) & (before_owner != torn_cell)).sum())
    torn = int(state.torn[0]) - torn_before
    touching, opened = axiom_contacts(state)
    web = stats(state, settings)["web_pieces"]
    print(f"split at step {split_at}; energy volumes {before_volumes} -> {volumes_after}; torn cells {torn_cells}, "
          f"biggest volume before {biggest} with {held[biggest]} boxes; split-marked boxes that "
          f"belonged to other cells {others_marked}; tear {torn} quanta, minus rose {after['minus'] - before['minus']}; "
          f"touching {touching}, open {opened}, web pieces {web}")
    assert volumes_after == before_volumes + 1
    assert torn_cell == biggest and held[biggest] >= settings.split_across ** 3 and others_marked == 0
    assert torn > 0 and after["minus"] - before["minus"] == torn
    assert touching == 0 and opened == 0 and web == 1


@needs_gpu
def test_no_joint_starts_sealed_and_none_collapses_in_20_steps():
    settings = Settings()
    state = new_universe()
    start_squeeze = stats(state, settings)["squeeze_max"]
    for _ in range(20):
        step(state, settings)
    squeeze = stats(state, settings)["squeeze_max"]
    collapsed = int(state.collapses[0])
    print(f"squeeze max at the start {start_squeeze:.3f}, after 20 steps {squeeze:.3f}, collapsed {collapsed}")
    assert start_squeeze < settings.seal_squeeze
    assert collapsed == 0


@needs_gpu
def test_a_tear_keeps_one_web_every_volume_wrapped_and_no_joint_collapses_after_it():
    settings = Settings()
    boxes = host_boxes(new_universe(5, 10, seed=1, unevenness=0.0, margin=10))
    quanta = boxes["quanta"].copy()
    quanta[boxes["owner"] == V] = 100
    graph = {name: boxes[name] for name in ("owner", "kind", "between", "place", "size", "size_energy")}
    state = State.from_host(quanta, np.zeros(quanta.shape, dtype=np.float32), seed=1, lattice=graph)
    rows = []
    before_split = 0
    for count in range(1, 301):
        if state.splits[0] == 0:
            before_split = int(state.collapses[0])
        step(state, settings)
        if state.splits[0] > 0:
            touching, opened = axiom_contacts(state)
            rows.append((count, int(state.collapses[0]), stats(state, settings)["web_pieces"], touching, opened))
            if len(rows) == 15:
                break
    print(f"collapses before the split {before_split}; after the first split (step, collapses so far, web pieces, "
          "touching, open): " + "; ".join(" ".join(str(value) for value in row) for row in rows))
    assert len(rows) == 15
    assert rows[-1][1] == before_split
    assert all(row[2] == 1 and row[3] == 0 and row[4] == 0 for row in rows)
