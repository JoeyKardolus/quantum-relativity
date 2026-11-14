"""What actually happened in a recording, in plain sentences built from its numbers.

Contract: `what_happened(recording)` returns two or three sentences. The first:
the steps run, the joint count from first to last frame and the step joints
first collapsed (or that none did), and the far background's mean connection
length relative to the start (the size of space) at the last frame that still
has a far background, saying when it is gone. The second, when two joints are
marked: whether the lumps got closer relative to the space around them by that
same frame (their gap in far-background connection lengths changed by more
than CHANGE either way, or not), with that gap and the raw gap in starting
mean lengths at both ends. A third gives the rules' message when the
recording stopped early. Only numbers from the frames are used.
"""
from __future__ import annotations

import math

from physics.viewer.recording import Recording

CHANGE = 0.03   # a relative gap change smaller than this counts as keeping the distance


def what_happened(recording: Recording) -> str:
    """Plain sentences from the recording's first and last frames."""
    first, last = recording.frames[0], recording.frames[-1]
    collapsed = next((frame.step for frame in recording.frames if frame.joint_count < first.joint_count), None)
    joints = (f"no joint collapsed, all {first.joint_count} stayed" if collapsed is None else
              f"joints first collapsed at step {collapsed} and went from {first.joint_count} to {last.joint_count}")
    readable = next(frame for frame in reversed(recording.frames) if math.isfinite(frame.background_length))
    size = readable.background_length / first.background_length
    lumps = "lump" if len(first.marked) == 1 else "lumps"
    space = (f"the space far from the {lumps} ended at {size:.1%} of its starting size (mean connection length)"
             if readable is last else
             f"the space far from the {lumps} was at {size:.1%} of its starting size at step {readable.step} "
             "and then collapsed with the rest")
    sentences = [f"Over {last.step} steps {joints}, and {space}."]
    if first.marked_gap is not None and readable.marked_gap is not None:
        sentences.append(_lumps(first, readable))
    if recording.stopped:
        sentences.append(f"The run stopped early at {recording.stopped}.")
    return " ".join(sentences)


def _lumps(first, last) -> str:
    start = first.marked_gap / first.background_length
    end = last.marked_gap / last.background_length
    raw = (f"; the raw gap went from {first.marked_gap / first.mean_length:.2f} to "
           f"{last.marked_gap / first.mean_length:.2f} starting mean lengths")
    change = end / start - 1
    verdict = ("got closer" if change < -CHANGE else "moved apart" if change > CHANGE
               else "kept their distance")
    return (f"Relative to the space around them the lumps {verdict} by step {last.step}: their gap went from {start:.2f} "
            f"to {end:.2f} far-background connection lengths ({change:+.1%}){raw}.")
