<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 248. A flaky suite makes a generator irreproducible, and a seed without an instance need not fail admission

**Lesson:** a generator never decides on one run of something that can vary. The repository plugin keeps a mutant only
when the project's own suite fails one to six tests on it; under load one suite run (the Lua-hosted projects take about a
second) differed from another, so the same request gave different folders and the reproducibility gate rejected one of
the twelve environments of the 2026-11-13 compile ("generating the same request twice gave different folders").

**Fix:** every suite run the repository plugin decides on is made twice, and two runs that disagree count as a broken
suite (`agreed`, spec `two_runs_that_disagree_are_a_broken_suite_and_two_that_agree_are_the_result`). A seed for which a
plugin then finds no valid instance is skipped and the next seed of the same stream drawn, at most eight per instance
asked, and counted in the admission report (`skipped_seeds`); a plugin that never produces one still fails admission
(`a_seed_the_plugin_cannot_make_an_instance_for_is_skipped_and_counted_not_fatal`).

**Also seen in that compile:** the host runner's 8 GiB address-space cap silenced the emulator (it maps guest memory and
a translation arena up front and printed nothing). The cap is 64 GiB and the firmware check reports what it read when it
is empty; both were already on main when this entry was written.
