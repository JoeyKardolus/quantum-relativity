# The score, logic

- `score_samples.py`: the score of a run's samples, its two halves, the hold factor
  and every part.

## Must not get wrong

- **The first sample is the start.** Every gain is measured against it.
- **Without two lumps there is no gap**, and the gap's half of the expansion part is 0.
- **Fewer than four samples, or an observer's time that stands still, raises.** The
  search turns that into a recorded 0 with the reason; do not hide it here.
