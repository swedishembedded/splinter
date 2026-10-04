# 003. Training and evaluation must render the same prompt

A reasoning model whose chat template opens a think block in the prompt was
fine-tuned on answers that never followed the empty closed block it is asked
from, so two runs were scored on a prefix they never trained on and their
held-out loss did not predict exam behaviour. The rewrite that adds the block
skipped any record with a `tools` key, and every dataset here writes
`"tools": []`: the check was for presence, not for a non-empty list. The
trained file was byte-identical to the input, which is how it was found.

Check by diffing the prepared training file against the input, and by
asserting that a training rendering of an assistant turn starts with exactly
the bytes the evaluation prompt ends with. Preference pairs need the same
treatment: the chat template drops a closed think block from a turn it
renders as history unless the trainer keeps it.
