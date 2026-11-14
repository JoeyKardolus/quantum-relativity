"""The gravity test's result dict: verdict, numbers, summary and meaning.

Contract: `gravity_result(histories, stopped, heavy, bending, plots, picture,
settings, seed, seconds, ruler)` judges the runs and returns the
JSON-serialisable result dict.

The verdict rule is fixed here, before any run, and answers the milestone's
question (attraction off): the lumps "approach" when their size-relative centre
distance falls by more than APPROACH_THRESHOLD beyond the control's change; if
their peak excess has dropped below IDENTITY_FLOOR of its start, the lumps have
dissolved and the verdict is "unclear" whatever the distance did. An approach
counts as a "pass" only if the lump centres also travel toward each other by
TRAVEL_THRESHOLD on a fixed ruler; an approach with centres that stay on the
same joints is the gap shrinking as energy spreads into it, and is "unclear".
The attraction-on run is judged by the same approach and travel thresholds and
reported in the numbers. A heavy lump "runs away" when its run breaks down or
loses more than RUNAWAY_SHARE of the joints within the run; it runs away "on
its own" when that happens before the two-lump attraction-on run (same rules,
ordinary lumps) has lost a tenth of its joints. Because a runaway
distorts every distance, the attraction-on approach is also reported at the
last step that still had HELD_SHARE of its joints.
"""
from __future__ import annotations

from physics.experiments.gravity.curves import lump_curves

APPROACH_THRESHOLD = 0.03
TRAVEL_THRESHOLD = 0.5   # fixed-ruler centre movement (mean lengths) that counts as travel
IDENTITY_FLOOR = 0.25
RUNAWAY_SHARE = 0.5
HELD_SHARE = 0.9         # "before the runaway": the last step with this share of joints left
V1_ON_COLLAPSE_STEP = 39   # v1 page, seed 1: attraction on, nothing resisting


def gravity_result(histories: dict, stopped: dict, heavy: list[dict], bending: dict,
                   plots: list[str], picture: str, settings: dict, seed: int,
                   seconds: float, ruler) -> dict:
    """Judge the runs and build the result dict."""
    curves = {label: lump_curves(records, ruler) for label, records in histories.items()}
    off, on, control = (curves[label] for label in ("attraction off", "attraction on", "control, no lumps"))
    counts = [record["joints"] for record in histories["attraction on"]]
    held = max(index for index, count in enumerate(counts) if count >= HELD_SHARE * counts[0])
    facts = {
        "off_change": float(off["centre"][-1] - 1), "on_change": float(on["centre"][-1] - 1),
        "joint_off": float(off["joint"][-1] - 1), "control_change": float(control["joint"][-1] - 1),
        "shift_off": float(off["fixed"][-1] - off["fixed"][0]),
        "shift_on": float(on["fixed"][-1] - on["fixed"][0]),
        "on_steps": len(histories["attraction on"]) - 1, "steps": settings["steps"],
        "on_stopped": stopped["attraction on"],
        "on_size": histories["attraction on"][-1]["scale"] / histories["attraction on"][0]["scale"],
        "on_joints_lost": histories["attraction on"][0]["joints"] - histories["attraction on"][-1]["joints"],
        "peak_kept": histories["attraction off"][-1]["lumps"][0]["peak"]
                     / histories["attraction off"][0]["lumps"][0]["peak"],
        "heavy": [_heavy_facts(run) for run in heavy], "bending": bending,
        "held": held, "held_change": float(on["centre"][held] - control["joint"][held]),
        "held_shift": float(on["fixed"][held] - on["fixed"][0]),
        "lump_pull_ratio": settings["lump_pull_ratio"],
    }
    facts["on_approach"] = facts["on_change"] - facts["control_change"] < -APPROACH_THRESHOLD
    facts["on_travelled"] = facts["shift_on"] < -TRAVEL_THRESHOLD
    verdict = _verdict(facts)
    return {"milestone": 3, "title": "Gravity test",
            "question": "With the attraction rule off, do two lumps of energy move toward "
                        "each other from the network's shape alone? And with the v2 pull and "
                        "joint push on, do they approach, and does a heavy lump collapse?",
            "verdict": verdict, "summary": _summary(verdict, facts), "meaning": _meaning(facts),
            "numbers": _numbers(facts, settings), "plots": plots, "picture": picture,
            "settings": settings, "seed": seed, "seconds": round(seconds, 1)}


def _heavy_facts(run: dict) -> dict:
    start, end = run["joints"][0], run["joints"][-1]
    runaway = run["stopped"] is not None or end < (1 - RUNAWAY_SHARE) * start
    halved = [index for index, count in enumerate(run["joints"]) if count < (1 - RUNAWAY_SHARE) * start]
    return {"amplitude": run["amplitude"], "start": start, "end": end,
            "half_step": halved[0] if halved else (run["stopped"]["step"] if run["stopped"] else None),
            "steps": len(run["joints"]) - 1, "stopped": run["stopped"], "runaway": runaway,
            "first_step_lost": start - run["joints"][1] if len(run["joints"]) > 1 else None}


def _verdict(facts: dict) -> str:
    approach = facts["off_change"] - facts["control_change"] < -APPROACH_THRESHOLD
    travelled = facts["shift_off"] < -TRAVEL_THRESHOLD
    if facts["peak_kept"] < IDENTITY_FLOOR or (approach and not travelled):
        return "unclear"
    return "pass" if approach else "fail"


def _numbers(facts: dict, settings: dict) -> list[dict]:
    on_rules = settings["attraction_on_rules"]
    numbers = [
        {"name": "attraction off: change in size-relative lump-centre distance",
         "measured": facts["off_change"], "known": None,
         "note": f"approach needs < control - {APPROACH_THRESHOLD}"},
        {"name": "attraction off: change in size-relative starting-joint distance",
         "measured": facts["joint_off"], "known": None,
         "note": "same, but the fixed starting joints instead of tracked centres"},
        {"name": "control (no lumps): change in size-relative joint distance",
         "measured": facts["control_change"], "known": 0.0, "note": "nothing should move without lumps"},
        {"name": "attraction off: lump-centre separation change on a fixed ruler (mean lengths)",
         "measured": facts["shift_off"], "known": None,
         "note": "0 means the centres stayed on the same joints: the gap shrank, nothing travelled"},
        {"name": "attraction off: lump A peak excess kept at the end (share of start)",
         "measured": facts["peak_kept"], "known": None,
         "note": f"below {IDENTITY_FLOOR} counts as dissolved"},
        {"name": "attraction on: steps the network stayed readable",
         "measured": facts["on_steps"], "known": None,
         "note": f"{facts['steps']} = the whole run; v1 (no joint push) broke at step {V1_ON_COLLAPSE_STEP}"
                 + (f"; stopped: {facts['on_stopped']['reason']}" if facts["on_stopped"] else "")},
        {"name": "attraction on: network size, end / start", "measured": facts["on_size"], "known": None,
         "note": "median network distance from 8 fixed joints"},
        {"name": "attraction on: joints lost to joint collapse", "measured": facts["on_joints_lost"],
         "known": None, "note": f"push limit {on_rules['push_limit']:.4g}, joint push {on_rules['joint_push']:g}"},
        {"name": "attraction on: change in size-relative lump-centre distance",
         "measured": facts["on_change"], "known": None,
         "note": f"approach needs < control - {APPROACH_THRESHOLD}; "
                 + ("approached" if facts["on_approach"] else "no approach")},
        {"name": "attraction on: lump-centre separation change on a fixed ruler (mean lengths)",
         "measured": facts["shift_on"], "known": None,
         "note": f"travel needs < -{TRAVEL_THRESHOLD}"},
        {"name": f"attraction on, step {facts['held']} (last with {HELD_SHARE:.0%} of joints): "
                 "size-relative centre distance change minus the control's",
         "measured": facts["held_change"], "known": None,
         "note": f"approach needs < -{APPROACH_THRESHOLD}; before the runaway distorts distances"},
        {"name": f"attraction on, step {facts['held']}: lump-centre separation change on a fixed ruler",
         "measured": facts["held_shift"], "known": None, "note": f"travel needs < -{TRAVEL_THRESHOLD}"},
        {"name": "pull at a lump centre / median background pull, at the start",
         "measured": facts["lump_pull_ratio"], "known": None,
         "note": "how much a lump stands out; near 1 means the pull hardly fades with distance"},
    ]
    for run in facts["heavy"]:
        ended = (f"broke down at step {run['stopped']['step']}: {run['stopped']['reason']}"
                 if run["stopped"] else f"ran all {run['steps']} steps")
        own = run["half_step"] is not None and run["half_step"] <= facts["held"]
        numbers.append({"name": f"heavy lump, amplitude {run['amplitude']:g}: joints left at the end",
                        "measured": run["end"], "known": None,
                        "note": f"of {run['start']}; {run['first_step_lost']} lost in step 1; {ended}; "
                                + ("runs away on its own" if own else
                                   "runs away only with the whole-network squeeze" if run["runaway"]
                                   else "does not run away")})
    bending = facts["bending"]
    numbers += [
        {"name": "light bending: mean closest approach of paths to a lump, before",
         "measured": bending["mean_before"], "known": None,
         "note": f"{bending['pairs']} far joint pairs, fixed flat ruler, mean lengths"},
        {"name": "light bending: mean closest approach, after the lump settles",
         "measured": bending["mean_after"], "known": None,
         "note": f"{bending['closer']} paths moved closer, {bending['farther']} farther; the "
                 "lump's short connections act as a shortcut, whereas in general relativity "
                 "space near a mass is stretched, so the sign matches but the mechanism differs"}]
    return numbers


def _summary(verdict: str, facts: dict) -> str:
    motion = "did not get closer" if verdict == "fail" else "got closer"
    travel = ("the lumps themselves stayed put and the gap between them shrank as their energy "
              "spread into it" if abs(facts["shift_off"]) < TRAVEL_THRESHOLD else "the lump centres moved")
    return (f"With the attraction rule off the two lumps {motion} (distance change "
            f"{facts['off_change']:+.1%} against {facts['control_change']:+.1%} in an empty control), "
            f"and their centres moved {facts['shift_off']:+.2f} connection lengths on a fixed ruler, so "
            f"{travel}. {_meaning(facts)}")


def _meaning(facts: dict) -> str:
    lost = facts["on_joints_lost"]
    joints = facts["heavy"][0]["start"]
    if facts["on_stopped"] is None and lost < (1 - HELD_SHARE) * joints:
        held = (f"With pull and joint push on, the network stayed whole for all {facts['steps']} steps "
                f"(v1, with nothing pushing back, broke at step {V1_ON_COLLAPSE_STEP})")
    else:
        end = (f"broke down at step {facts['on_steps'] + 1}" if facts["on_stopped"]
               else f"lost {lost} of {joints} joints")
        held = (f"With pull and joint push on, the network held out longer than v1 but still {end} "
                f"(v1, with nothing pushing back, broke at step {V1_ON_COLLAPSE_STEP}), as joint "
                f"collapse spread until {lost} of {joints} joints had merged")
    approach = facts["held_change"] < -APPROACH_THRESHOLD and facts["held_shift"] < -TRAVEL_THRESHOLD
    lumps = (f"the two lumps moved toward each other ({facts['held_change']:+.1%} against the control, "
             f"{facts['held_shift']:+.2f} lengths on a fixed ruler, at step {facts['held']})" if approach
             else f"up to step {facts['held']} the two lumps did not approach "
                  f"({facts['held_change']:+.1%} against the control, {facts['held_shift']:+.2f} lengths "
                  "on a fixed ruler)")
    own = [run for run in facts["heavy"] if run["half_step"] is not None and run["half_step"] <= facts["held"]]
    later = [run for run in facts["heavy"] if run not in own]
    parts = [f"amplitude {run['amplitude']:g} took half the network within {run['half_step']} step{'' if run['half_step'] == 1 else 's'}"
             for run in own]
    parts += [f"amplitude {run['amplitude']:g} lost {run['first_step_lost']} joints at once and then held"
              + (f" until the whole-network squeeze took it at step {run['half_step']}"
                 if run["half_step"] is not None else "") for run in later]
    heavy = ("A single heavy lump past the push limit set off joint collapse that ran away on its own"
             if own else "No single heavy lump past the push limit set off a runaway of its own") \
        + f" ({'; '.join(parts)})"
    if approach:
        reading = "the pull drew the lumps together before anything else gave way."
    else:
        squeezed = ("the joint push delays that squeeze but does not stop it"
                    if facts["on_stopped"] or lost >= (1 - HELD_SHARE) * joints
                    else "the joint push held that squeeze off for the whole run")
        reading = (f"a lump's pull is only {facts['lump_pull_ratio']:.2f}x the background pull, so "
                   "the pull barely fades with distance and squeezes the whole network rather than "
                   f"drawing lumps together; {squeezed}.")
    joined = "as rule 8 says, but" if own else "and"
    return f"{held}, and {lumps}. {heavy}, {joined} {reading}"
