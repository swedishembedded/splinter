<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 118. The cpt view wrote a corpus no trainer read

`dataset build --view cpt` projected the text of sources but wrote Splinter's export format, which
brain refuses, so continued pretraining had a view and no way to train. A text is now written as one
supervised assistant turn with nothing before it (the next-token loss over every token of it), which
`train` reads as any chat dataset. On the toy checkpoint a raw document of cipher lines `abc=bcd` took
held-out exact match from 0.00 to 0.52, but only after 3000 steps (500 gave 0.03, 1200 gave 0.20):
three of every eight tokens of a line are unpredictable, so the loss on the learnable ones is diluted.

Caveat: the chat template still frames the turn, so this is not pretraining on bare text.
