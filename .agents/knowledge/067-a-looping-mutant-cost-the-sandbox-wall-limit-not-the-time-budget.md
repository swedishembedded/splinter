<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 067. A looping mutant cost the sandbox's wall limit, not the generator's time budget

The generator refuses a mutant whose suite runs past a time budget (the larger of a
multiple of the baseline and 60 s), but it only checked the budget after the run: the run
itself was bounded by the workspace host's command limit of 300 s. Python families whose
mutants loop forever (the suites have no per-test timeout) therefore cost five minutes
each. Measured on the mdrender family with the same flags as the other funnels: a run of
more than two hours had written 78 of about 150 artifacts, with a `python3 -m unittest`
process 7 minutes old at 100% CPU when inspected. After the suite is run under `timeout`
with the budget (exit status 124 counted as over the budget), the same family finished
in about 45 minutes with 15 mutants refused as `over_time_budget`.

How to avoid it: a limit checked after the work does not bound the work. Pass the bound to
the process that does it.
