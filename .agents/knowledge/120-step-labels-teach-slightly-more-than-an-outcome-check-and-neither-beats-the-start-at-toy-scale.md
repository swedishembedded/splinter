<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 120. Step labels teach slightly more than an outcome check, and neither beats the start at toy scale

A two-step task (apply the cipher, then apply it to your own first answer), a checker that judges each
step in the context the conversation gave it (`splinter_data::process`), and two arms trained from the same
720 rollouts: the outcome arm keeps the 19 to 30 conversations whose every step was right, the step arm
keeps the 270 to 340 good steps. Scored on 200 unseen prompts, the model they started from scored 0.15,
the outcome arm 0.155 and the step arm 0.185. Earlier configurations gave step above outcome by 0.04 to 0.05
(0.21 vs 0.16; 0.10 vs 0.05), always within a few prompts of each other and of the start; a learning
rate of 5e-3 took both below the start.

Rule: the data kept differs by an order of magnitude, the score does not yet: treat process verification
as built and not shown. It needs more rollouts per prompt and a stronger start than a CPU smoke affords.
The row of `make test/methods` says NOT SHOWN.
