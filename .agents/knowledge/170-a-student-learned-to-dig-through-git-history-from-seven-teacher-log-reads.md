<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 170. A student learned to dig through git history from seven teacher log reads

**Lesson:** a demonstration teaches its exploration, and a small student generalises it past the
line the leak audit draws. A teacher that read `git log --oneline -20` and `git show --stat HEAD`
(allowed: the checked-out commit's own history) taught a 4B student `git log --all`, `git reflog`
and reads of `.git/refs` and `.git/objects` (failed: history beyond HEAD, raw storage). Leaving out
the trials the audit flagged is not enough; curate the precursor reads out of the demonstrations.

Found 2026-11-18 in pilot A2 (Qwen3-4B SFT on 420 packed teacher trajectories, splinter 3fc94481f):
the base model had no security violation on workspace-tools-reference; the three A2 seeds had 4 to 10
each (and 3 on compiled dev, seed 3), all git history probing. The dataset supervised 7 history reads
in 7 of 420 records, every one `git log --oneline -N` or `git show --stat HEAD`. Since then the
sft-step view drops such actions by default (`splinter_data::curation`, policy
`no-repository-history`): on the pilot's store it drops 14 of 3522 supervised actions (14 of 445
trajectories). Whether that removes the learned probing is a training measurement not yet made.
