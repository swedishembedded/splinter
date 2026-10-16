# 049. A gate that could not grade its canaries is not clean

A canary check has two ways to look clean while measuring nothing: the
candidate's attempt at a canary can be set aside (the harness could not run
it, the exposure audit failed it), and the whole suite can fail to grade (the
model did not load for the second time in one gate). Counting either as "no
canary passed" would release a candidate on a check that never ran.

The check therefore counts tasks and graded tasks apart: a withheld canary is
counted, not passed, and a gate handed canaries of which none was graded is
unmeasured and blocks the release, as every other unmeasured check does. In
the pipeline fixture, a candidate that answers one of three canaries and
passes every other check is refused with that canary's address named; one that
answers none is released with the check recorded (2 of 2 graded, none passed);
a gate handed none has no canary entry at all.

Two decisions worth knowing: canaries are attempted by the candidate only (a
canary carries no comparison, and grading the champion would cost a second pass
over a suite that cannot move the decision), and any single pass refuses (the
research plan's 5% rate needs a canary population large enough to have a rate;
with the handful a gate is handed today, one pass is already the evidence).

How to avoid it: when adding a check to the gate, decide first what an
ungraded input does, and make it the failing answer.
