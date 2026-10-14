# 025. An audit joined on a name that repeats

The extraction audit of the synthetic-author sample compared each decision read
out of an essay with what the beliefs dictate for "the case with this name".
Case names are drawn from a small stock and repeat from one article to another,
so a decision was compared with another article's case of the same name and 8
to 10 percent of them looked wrong. Joined on the article and the case, 10 of
416 disagreed. The paper nearly carried the wrong figure.

How to avoid it: a join key is unique or it is not a key. Where a name can
repeat, join on the pair that cannot, and assert uniqueness of the key when the
lookup is built. A surprising error rate in a validation step is a reason to
audit the audit before reporting it.
