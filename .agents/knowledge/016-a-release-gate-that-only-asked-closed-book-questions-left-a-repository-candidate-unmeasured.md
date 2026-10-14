# 016. A release gate that only asked closed-book questions left a repository candidate unmeasured

A candidate trained by `practice` reached the release gate on every run and was
never released. The gate's suites were built from the records training held out;
a record's task was admitted to a suite only if it was closed-book, so every
task of a change to a repository was left out and counted under `not_closed_book`.
The improvement suite held no task, the check read `not measured: no held-out
task ... was graded for both models`, and the serve check could only have
re-asked closed-book questions through the plain endpoint. The gate failed
loudly, as it is meant to when a check cannot be measured, and permanently:
nothing the candidate did could change the outcome, and the pipeline's last
stage had never once produced a release.

The gate now probes tasks that are changes to a repository the way the solve
stage attempts them: the worker in a copy of the task's snapshot, the same
bounds and per-attempt seeds for the champion, the candidate and the served
candidate, graded by the workspace verifier and the exposure audit. The tasks
are the exam the candidate's run reserved before any attempt (recorded against
its dataset, so every earlier release is held to the exam of the run that made
it), clustered by family. The scripted spec has a candidate that mends five of
five exam families the champion leaves alone: the sign test counts five
discordant families, p = 1/32, and the candidate is released; a candidate no
better than the champion has no discordant family and is blocked with that
reason; one that does worse than the champion on an earlier release's exam is
blocked by retention. Attempts that read the hidden tests or that the harness
could not run are set aside and counted per arm; they are in no denominator.

How to avoid it: a pipeline whose last stage has never succeeded is untested at
that stage. A path that produces candidates of a kind needs a probe for that kind
before it is given a release step, and a spec that ends with a release, not with
"reaches the gate".
