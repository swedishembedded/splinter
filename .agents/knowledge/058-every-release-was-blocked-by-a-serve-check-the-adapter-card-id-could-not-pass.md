# 058. Every release was blocked by a serve check its adapter's card id could not pass

The release gate's serve check starts `brain serve --adapter <candidate>` and
reports "brain serve exited (exit status: 1) before it served", followed by a
screen of unrelated skipped-model notices that hides the cause. Run by hand, the
last line was: `adapter id "local/Qwen3-0.6B:splinter:candidate" is not a model
reference (... exactly ":owner:name:tag" ...)`. Training recorded the card id as
`<base>:splinter:candidate`, one segment short of brain's adapter grammar, so
the serve check failed for every candidate and no release could be made, whatever
the improvement, retention and anchor checks said. The candidate id now ends in a
tag.

How to avoid it: when a check "could not be measured", reproduce its command by
hand and read the end of its output, not the beginning; and give each handoff
between projects (an id written here, parsed there) a test against the grammar
the reader enforces.
