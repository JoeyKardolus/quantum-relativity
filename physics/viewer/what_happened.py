"""What actually happened in a recording, in plain sentences built from its numbers.

Contract: `what_happened(recording)` returns one or two sentences: the steps
run, the joint count from first to last frame and the step joints first
collapsed (or that none did), the gap between the first two marked joints in
mean connection lengths when there are two, the mean connection length, and
the rules' message when the recording stopped early. Only numbers from the
frames are used; nothing is interpreted beyond them.
"""
from __future__ import annotations

from physics.viewer.recording import Recording


def what_happened(recording: Recording) -> str:
    """Plain sentences from the recording's first and last frames."""
    first, last = recording.frames[0], recording.frames[-1]
    collapsed = next((frame.step for frame in recording.frames if frame.joint_count < first.joint_count), None)
    joints = (f"no joint collapsed, all {first.joint_count} stayed" if collapsed is None else
              f"joints first collapsed at step {collapsed} and went from {first.joint_count} to {last.joint_count}")
    sentence = (f"Over {last.step} steps {joints}; the mean connection length went from "
                f"{first.mean_length:.4f} to {last.mean_length:.4f}")
    if first.marked_gap is not None and last.marked_gap is not None:
        sentence += (f", and the gap between the lump centres went from "
                     f"{first.marked_gap / first.mean_length:.1f} to "
                     f"{last.marked_gap / last.mean_length:.1f} mean connection lengths")
    sentence += "."
    if recording.stopped:
        sentence += f" The run stopped early at {recording.stopped}."
    return sentence
