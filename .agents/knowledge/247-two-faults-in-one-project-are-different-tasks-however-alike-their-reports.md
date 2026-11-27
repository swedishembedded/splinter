<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 247. Two faults in one project are different tasks however alike their reports

**Lesson:** the text an environment's novelty is judged on is what distinguishes its task (the changed lines, the drawn
identifiers), never only what the solver is told. The first environment-factory compile on real projects (2026-11-13)
rejected the easiest rungs for novelty: at the lowest detail the report is one sentence ("tests in X fail", "the console
output differs"), so two different injected faults in one project produced the same instruction and the novelty gate
called each later instance an exact repeat. Three of the twelve environments of that compile were rejected this way:
the repository environment's easy rung, the firmware environment's easy rung, and the database workflow's easiest rung,
whose order number was a function of the request index alone.

**Fix:** the novelty text of a fault-injected instance is its symptoms plus the lines the fault changed
(`factory::plugin::fault_novelty_text`, from the faulty files the plugin records in `solution/changed.txt`), and a
database workflow's order number carries a drawn part. The plugin versions moved (repository-mutants 2,
firmware-emulator 2, db-workflow 2). Spec: `two_faults_with_the_same_report_have_different_novelty_texts`.
