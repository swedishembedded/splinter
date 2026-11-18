<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 169. A dataset in ingest order held out and monitored one domain

**Lesson:** a held-out rule that takes the newest records of a file is a property of the file's
order, not of the data. A bench export lists its trials domain by domain, so the newest tenth of
an sft-step dataset built from it is all the last domain. Draw held-out and monitoring sets by a
recorded seed within each stratum, so they hold every domain at its share wherever the records sit.

Found 2026-11-18 in pilot A2 (SFT of Qwen3-4B on 420 verified teacher trajectories of four bench
domains, splinter 3fc94481f): built from the export as ingested, every held-out and monitoring
record was accounting (the last domain), which left 4 of 83 accounting records in training and
scored the run's held-out loss and its step selection on one domain. The pilot shuffled the export's
index (seed 20261118) before ingest and trained again (`logs/aborted-tailsplit`). Since then
`dataset build` writes a `split_draw` (seed, stratum = the first part of the bench task's
`dimension`) on every record and records the split in the manifest; the holdout rule orders groups by
a hash of the seed within each stratum and interleaves the strata by share. On the pilot's layout
(179/82/76/83 records) it holds out each domain at its share within one record, 42 in all (`holdout::tests::a_seeded_split_holds_out_every_stratum_at_its_share_of_the_set`).
Records without a draw keep the newest-end rule, so a candidate trained before is gated on what it
held out.
