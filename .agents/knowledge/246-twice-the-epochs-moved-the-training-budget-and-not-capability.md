<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 246. Twice the epochs moved the training budget and not capability

**Lesson:** a knob is proven by the state variable it moves, and a knob that moves its variable can still leave
capability where it was. The Qwen3-4B generation-2 candidate of the source-to-transfer loop trained 0.97 epochs at its
selected step under the 800-step cap (knowledge 235). Retrained with the epoch budget at 1.89 (generation 1's
effective epochs), everything else unchanged, the epoch knob moved the training budget from 0.97 to 1.89 epochs, and
capability did not follow: against the same champion, the retrained candidate passed 6 of 399 exam attempts (the
champion 11 of 400) and 53 of 260 own held-out attempts (the champion 57 of 264); the 0.97-epoch candidate had passed
13 of 400 and 66 of 264.

So the cap was not what stalled generation 2. What the observer reads at that generation (knowledge 244) is the other
candidate: about 40% of the records are the champion's own, none of them from the tasks it fails.

**How to apply:** declare each recipe knob with the variable it moves (`knobs:`, `splinter probe`) and probe it on a
cheap slice before a loop leans on it: a moved variable says the knob works, the capability rows say whether that
variable is the one that limits the loop. Here the next knob to probe is the frontier (which tasks' records are kept).
