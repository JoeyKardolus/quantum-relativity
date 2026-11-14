"""The gravity test's result dict: verdict, numbers, summary and meaning.

Contract: `gravity_result(histories, stopped, heavy, plots, picture, settings,
seed, seconds, ruler, unit)` judges the runs and returns the JSON-serialisable
result dict. `histories` holds the records of the runs labelled MAIN, CONTROL
and DRIFT_OFF.

The verdict rule is fixed here, before any run. The main run (all rules on,
balancing drift) "approaches" when its lump-centre distance, measured in
far-background mean connection lengths so a whole-space shrink or growth does
not count, falls by more than APPROACH_THRESHOLD beyond the control's change
(same rules, no lumps). Pass needs that approach; fail is no approach. The
verdict is "unclear" when the lumps dissolve (lump A's peak excess below
IDENTITY_FLOOR of its start), when the main run breaks down, or when it loses
more than RUNAWAY_SHARE_MAIN of its joints (then a runaway, not an approach,
decides the distances). The drift-off run (v2 way) is judged by the same
measure and reported beside it. A heavy lump "runs away" when its run breaks
down or ends with fewer than half its joints.
"""
from __future__ import annotations

import numpy as np

from physics.experiments.gravity.curves import lump_curves

MAIN = "all rules, balancing drift"
CONTROL = "control: no lumps"
DRIFT_OFF = "all rules, drift off (v2)"
APPROACH_THRESHOLD = 0.03
IDENTITY_FLOOR = 0.25
RUNAWAY_SHARE_MAIN = 0.1
HEAVY_RUNAWAY_SHARE = 0.5


def gravity_result(histories: dict, stopped: dict, heavy: list[dict], plots: list[str],
                   picture: str, settings: dict, seed: int, seconds: float, ruler: np.ndarray,
                   unit: float) -> dict:
    """Judge the runs and build the result dict."""
    curves = {label: lump_curves(records, ruler, unit) for label, records in histories.items()}
    facts = _facts(histories, stopped, curves, heavy, settings)
    verdict = _verdict(facts)
    return {"milestone": 3, "title": "Gravity test",
            "question": "With every rule on and a drift that balances the even background pull, do "
                        "two lumps of energy approach each other relative to the space around them, "
                        "and does a heavy lump collapse?",
            "verdict": verdict, "summary": _summary(verdict, facts), "meaning": _meaning(verdict, facts),
            "numbers": _numbers(facts), "plots": plots, "picture": picture,
            "settings": settings, "seed": seed, "seconds": round(seconds, 1)}


def _facts(histories: dict, stopped: dict, curves: dict, heavy: list[dict], settings: dict) -> dict:
    main, control, off = curves[MAIN], curves[CONTROL], curves[DRIFT_OFF]
    off_last = int(np.flatnonzero(np.isfinite(off["centre"]))[-1])
    start_joints = histories[MAIN][0]["joints"]
    return {
        "main_change": float(main["centre"][-1] - 1), "control_change": float(control["joint"][-1] - 1),
        "off_change": float(off["centre"][off_last] - 1), "off_last": off_last,
        "main_gap": (histories[MAIN][0]["centre_distance"] / histories[MAIN][0]["background_length"],
                     histories[MAIN][-1]["centre_distance"] / histories[MAIN][-1]["background_length"]),
        "main_raw": (float(main["raw"][0]), float(main["raw"][-1])),
        "main_background": float(main["background"][-1]),
        "control_background": float(control["background"][-1]),
        "off_background": float(off["background"][off_last]),
        "main_shift": float(main["fixed"][-1] - main["fixed"][0]),
        "main_step_one": float(main["centre"][1] - 1),
        "main_after_step_one": float(main["centre"][-1] / main["centre"][1] - 1),
        "step_one_lost": start_joints - histories[MAIN][1]["joints"],
        "peak_kept": histories[MAIN][-1]["lumps"][0]["peak"] / histories[MAIN][0]["lumps"][0]["peak"],
        "main_steps": len(histories[MAIN]) - 1, "steps": settings["steps"],
        "main_stopped": stopped[MAIN], "off_stopped": stopped[DRIFT_OFF],
        "joints": start_joints, "main_lost": start_joints - histories[MAIN][-1]["joints"],
        "off_end": histories[DRIFT_OFF][-1]["joints"], "off_steps": len(histories[DRIFT_OFF]) - 1,
        "drift": settings["balancing_drift"], "heavy": [_heavy_facts(run) for run in heavy],
    }


def _heavy_facts(run: dict) -> dict:
    start, end = run["joints"][0], run["joints"][-1]
    halved = [index for index, count in enumerate(run["joints"]) if count < HEAVY_RUNAWAY_SHARE * start]
    return {"amplitude": run["amplitude"], "start": start, "end": end,
            "half_step": halved[0] if halved else None, "steps": len(run["joints"]) - 1,
            "stopped": run["stopped"], "runaway": run["stopped"] is not None or bool(halved)}


def _verdict(facts: dict) -> str:
    if (facts["peak_kept"] < IDENTITY_FLOOR or facts["main_stopped"] is not None
            or facts["main_lost"] > RUNAWAY_SHARE_MAIN * facts["joints"]):
        return "unclear"
    return "pass" if facts["main_change"] - facts["control_change"] < -APPROACH_THRESHOLD else "fail"


def _numbers(facts: dict) -> list[dict]:
    numbers = [
        {"name": "balancing drift per step (calibration on the even background, no lumps)",
         "measured": facts["drift"], "known": None,
         "note": "the one calibrated setting; one step of even space keeps its total length"},
        {"name": "main: change in lump-centre distance in far-background mean lengths",
         "measured": facts["main_change"], "known": None,
         "note": f"approach needs < control - {APPROACH_THRESHOLD}"},
        {"name": "control (no lumps): change in starting-joint distance in far-background mean lengths",
         "measured": facts["control_change"], "known": 0.0, "note": "nothing should move without lumps"},
        {"name": "drift off (v2): change in lump-centre distance in far-background mean lengths",
         "measured": facts["off_change"], "known": None,
         "note": f"at step {facts['off_last']}, the last with a readable far background"},
        {"name": "main: lump-centre distance in far-background mean lengths, start and end",
         "measured": list(facts["main_gap"]), "known": None, "note": "start is short: lump connections hold more energy"},
        {"name": "main: raw lump-centre distance in starting mean lengths, start and end",
         "measured": list(facts["main_raw"]), "known": None, "note": "counts any whole-space shrink"},
        {"name": "far-background mean connection length, end / start: main, control, drift off",
         "measured": [facts["main_background"], facts["control_background"], facts["off_background"]],
         "known": 1.0, "note": "1 means space kept its size"},
        {"name": "main: change in lump-centre distance in far-background mean lengths, step 1 alone",
         "measured": facts["main_step_one"], "known": None,
         "note": f"{facts['step_one_lost']} joints collapsed in step 1"},
        {"name": "main: change in lump-centre distance in far-background mean lengths, step 1 to the end",
         "measured": facts["main_after_step_one"], "known": None, "note": "the closing after the first collapse"},
        {"name": "main: lump-centre separation change on a fixed ruler (mean lengths)",
         "measured": facts["main_shift"], "known": None,
         "note": "0 means the centres stayed on the same joints and the gap itself shrank"},
        {"name": "main: lump A peak excess kept at the end (share of start)",
         "measured": facts["peak_kept"], "known": None, "note": f"below {IDENTITY_FLOOR} counts as dissolved"},
        {"name": "main: joints lost to joint collapse", "measured": facts["main_lost"], "known": None,
         "note": f"of {facts['joints']}; more than {RUNAWAY_SHARE_MAIN:.0%} makes the verdict unclear"},
        {"name": "drift off (v2): joints left at the end", "measured": facts["off_end"], "known": None,
         "note": f"of {facts['joints']} after {facts['off_steps']} steps"
                 + (f"; stopped: {facts['off_stopped']['reason']}" if facts["off_stopped"] else "")},
    ]
    for run in facts["heavy"]:
        ended = (f"broke down at step {run['stopped']['step']}: {run['stopped']['reason']}"
                 if run["stopped"] else f"ran all {run['steps']} steps")
        numbers.append({"name": f"heavy lump, amplitude {run['amplitude']:g}: joints left at the end",
                        "measured": run["end"], "known": None,
                        "note": f"of {run['start']}; {ended}; "
                                + (f"half gone by step {run['half_step']}" if run["half_step"] is not None
                                   else "never lost half")})
    return numbers


def _summary(verdict: str, facts: dict) -> str:
    return (f"Verdict {verdict}: with every rule on and the balancing drift {facts['drift']:.5f} per step, "
            f"the lump-centre distance in far-background mean lengths changed {facts['main_change']:+.1%} "
            f"(from {facts['main_gap'][0]:.2f} to {facts['main_gap'][1]:.2f}), the control's "
            f"{facts['control_change']:+.1%}, and the v2 run without drift {facts['off_change']:+.1%}. "
            f"The far background ended at {facts['main_background']:.3f} of its start length with lumps, "
            f"{facts['control_background']:.4f} without, and {facts['off_background']:.3f} without drift; "
            f"the lump centres moved {facts['main_shift']:+.2f} mean lengths on a fixed ruler, and "
            f"{facts['main_lost']} of {facts['joints']} joints collapsed in the main run.")


def _meaning(verdict: str, facts: dict) -> str:
    if verdict == "pass":
        lumps = (f"With every rule on and a drift that balances the even background pull, the two lumps "
                 f"closed in on each other relative to the space around them (gap {facts['main_change']:+.0%} "
                 f"in far-background connection lengths, against {facts['control_change']:+.1%} with no "
                 f"lumps), but {'most of that was' if facts['main_step_one'] < facts['main_change'] / 2 else 'only part of that was'} "
                 f"step 1 alone ({facts['main_step_one']:+.0%}), when {facts['step_one_lost']} joints "
                 f"collapsed into the lumps, and from step 1 to the end the gap changed a further "
                 f"{facts['main_after_step_one']:+.0%}.")
    elif verdict == "fail":
        lumps = (f"With every rule on and a drift that balances the even background pull, the two lumps did "
                 f"not close in on each other relative to the space around them ({facts['main_change']:+.1%} "
                 f"against {facts['control_change']:+.1%} with no lumps).")
    else:
        lumps = (f"With every rule on and a balancing drift the result is unclear: the lump-centre distance "
                 f"changed {facts['main_change']:+.1%} against the space around it, but "
                 + ("the lumps dissolved" if facts["peak_kept"] < IDENTITY_FLOOR
                    else "joint collapse ran away") + ", so that number does not show an approach.")
    heavy = [run for run in facts["heavy"] if run["runaway"]]
    collapse = (f"a single heavy lump collapsed and ran away (amplitude {heavy[0]['amplitude']:g} took the "
                f"network from {heavy[0]['start']} joints to {heavy[0]['end']} in {heavy[0]['steps']} "
                f"step{'' if heavy[0]['steps'] == 1 else 's'})" if heavy else "no single heavy lump ran away")
    tipped = abs(facts["main_background"] - 1) > 0.01
    balance = (f"the balance, calibrated on even space, tips once lumps add pull (space far from them "
               f"ended at {facts['main_background']:.1%} of its size)" if tipped else
               f"the balance held with lumps too (space far from them ended at "
               f"{facts['main_background']:.1%} of its size)")
    return (f"{lumps} Without the drift the even pull squeezed all of space at once (the far background "
            f"fell to {facts['off_background']:.0%} by step {facts['off_last']} and {facts['off_end']} of "
            f"{facts['joints']} joints were left), {balance}, and {collapse}.")
