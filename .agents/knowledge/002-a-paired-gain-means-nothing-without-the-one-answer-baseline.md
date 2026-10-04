# 002. A paired gain means nothing without the one-answer baseline

A fine-tuned adapter looked like a large gain on a closed-choice question
(1% to 79% on a three-way attribution) and on recipient and year. It was
none of those: the adapter answered `A` to every question, and the held-out
scores equalled what one fixed answer scores. The base had been scoring low
only because it did not answer in the expected format, so the format was
counted as knowledge. The classes were skewed 77 to 5 to 15, and 500 steps
did not memorise passages (0% on continuations it had trained on).

A report over closed choices must print, beside every row, the share a single
answer for all questions would score, and the result per reference. A
benchmark of a few dozen scenarios needs the same floor for open answers: a
fixed template that always says "the method applies" and quotes the first
words of a passage scored 65% on a 26-scenario transfer benchmark, so a model
has to clear that, with an interval over the questions that excludes zero,
before any gain is claimed.
