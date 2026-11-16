<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 160. Brain's chat fine-tune of a from-scratch Ouro made it ignore its context; the raw-text objective on the same conversations did not

**Lesson:** a fine-tune whose loss falls while the probe falls is not learning the task; check what the
served checkpoint does on the training records themselves before tuning budgets. When a chat objective
degrades a model that the same data, rendered as text and trained with every token supervised, improves,
the defect is in the objective's path, not in the data.

Found 2026-11-16 in minimind on gpu3 with brain 16c2db09e (smoke profile, seed 7, 2.1M-param Ouro,
`samples/minimind`): the pretrained checkpoint solved 3 of 12 development tasks (it copies the instruction's
word into `grep -c ERROR`); `full_train` on the 5868 demonstration records (loss 1.51 -> 0.04 at lr 7e-4,
and 0.14 -> 0.24 at lr 1e-4 over 1000 steps) left 0 of 12, the model now running `grep -c TIMEOUT`
whatever the instruction said, and 0 of 96 own episodes verified. Served greedily on 12 of its own training
records it reproduced 7 raw continuations and fewer chat replies. The same demonstrations rendered in the
template and trained with `pretrain` (400 steps, 32 x 1024) kept 3 of 12 and verified 27 of 96 own
episodes; improving on those with `full_train` fell to 0 of 12, with `pretrain` rose to 4 of 12. The sample
trains its chat stages as text by default (`chat_training.as_text`); the `full_train` path for Ouro is
reported to brain. A record whose conversation holds a tool call with unparseable arguments is now left
out (`data::wire`), since brain refuses the whole dataset over one.

**Root cause (brain, 2026-11-16, ed9deabfa):** not the chat objective. The from-scratch checkpoints carried
`early_exit_threshold` 1.0, so serving always used the last loop pass, while the objective trains the pass
where the exit mass sits (pass 0, about 0.96-0.98 of it); `full_train` improved pass 0 and degraded the last
pass the server answered with. A from-scratch Ouro now initialises with 0.5 (the median exit). The sample
trains its fine-tune and improvement with `full_train` again (tool results unsupervised) and keeps
whole-trajectory text for the mid-training stage.
