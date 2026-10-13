<!--
SPDX-License-Identifier: CC-BY-4.0
Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
-->

# Measured results

`results.csv` is the only numerical source for the results tables. One row per
(experiment, repository, round, arm) comparison on frozen exam families, with the
paired outcome counts over family clusters (`wins`: adapter resolves, base does not;
`losses`: the reverse). No row exists until the run that produced it has finished;
`scripts/build_data.py` prints `NOT RUN` for an experiment without rows.
