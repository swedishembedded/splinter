# 048. A hidden suite that fails on the seeded fault passes its neighbours

Admission of a mutation-seeded workspace task asked only that the hidden tests
fail on the one fault and pass on the original. That is satisfied by a suite
that checks a single value: a plausible wrong patch of the neighbouring lines
passes it, so a pass said little. Measured on a two-function fixture with one
test file per function: every fault seeded in the neighbouring function was
caught by the visible file only, so the hidden file's kill rate was 0 of 2 to
3 (admission before the check admitted all of them); on the fixture whose one
test file covers its faults the rate was 2 of 2 and 0 variants for a file
with a single fault.

What the number is and is not:

- The denominator is the variants some test of the whole suite catches. A
  variant nobody catches may be equivalent to the original and cannot be told
  from a miss, so it is left out; zero such variants is no measurement
  (`kill_rate` is `None`), recorded as such and never as 0.
- A variant caught by the visible tests alone counts as a miss of the hidden
  tests, though the solver can see those tests fail. The floor (one half)
  is therefore a floor, not a target, and a task is refused only on at least
  two measured variants: one variant caught or missed is an anecdote.
- The cost is one suite run per variant (at most six) per admitted
  candidate, after every cheaper check.

How to avoid it: record the strength on the task, and let the gate weigh a
pass by it (open); calibrate the floor on tasks with seeded wrong patches.
