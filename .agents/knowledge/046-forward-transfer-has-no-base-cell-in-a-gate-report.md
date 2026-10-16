# 046. Forward transfer had no base cell to read in the gate reports

The continual-learning metrics read a results matrix A[k][i] from the gate
reports of a lineage. Backward transfer, forgetting and average final
performance fill from what the gates already record: each release's own suite
(improvement check) and every earlier suite (retention check). Forward
transfer and learning gain need the score of the untouched base model on each
suite, and only the first release's gate measures it, because its champion is
the base; every later gate's champion is the previous release.

Consequences, as measured on a fixture of two releases: the base cell exists
for suite 0 only, so forward transfer is typed "not measured" for every
suite after it, and the learning gain rests on suite 0 alone. Filling the base
cell from the champion would silently measure a different thing (progress over
the last release), and a zero would claim a measurement that was never made.

The other trap is the maximum in forgetting. Taking the maximum over all
generations, including the one before the task was learned, can credit a
score the model had before it learned the task (a base that already knew it);
the maximum runs from the generation that learned the task to the one before
the last. Backward transfer and forgetting differ exactly when a score peaked
after its learning step (fixture: -0.1 against 0.2).

How to avoid it: to get forward transfer on every suite, have the gate also
grade the base model on each new suite (one more arm in the probe), and keep
the cell absent where that was not done.
