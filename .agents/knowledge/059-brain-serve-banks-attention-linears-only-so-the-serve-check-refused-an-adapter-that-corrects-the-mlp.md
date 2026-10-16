# 059. `brain serve` banks the attention linears only, so the serve check refused an MLP-correcting adapter

After the card id was repaired (058) the gate's serve check
still failed for every candidate: "adapter ... corrects down, which this
server's --adapter-targets (wq,wk,wv,wo) does not; widen the flag to serve it".
Splinter's LoRA corrects every linear; brain's adapter bank covers the attention
linears unless `--adapter-targets` says otherwise. The check now starts the
server with `--adapter-targets all` whenever it offers an adapter, and the
test double of `brain serve` refuses an adapter offered without it, as brain does.

How to avoid it: a test double of a dependency must enforce the dependency's
refusals, not only its happy path; the double had accepted any flags, so the
suite stayed green while no real candidate could be served. The reason a check
could not be measured is the last line of the server's output, after the notices
of every model it skipped.
