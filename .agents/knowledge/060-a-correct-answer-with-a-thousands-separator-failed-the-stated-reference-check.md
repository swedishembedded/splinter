# 060. A correct answer with a thousands separator failed the stated-reference check

Of 28 tasks written from invented fact sheets, 9 were never taught: the teacher's
open-book answer failed verification. Reading three of them: "The population of
Raber is 4,711." against the reference "4711." failed, and so did "668 metres"
against "668 metres tall". The verifier already folded digit shapes and number
words, but not the commas that separate thousands, so a correct number written
in the usual way was a different number. A third of the facts were lost to
formatting before training began, and the same verifier grades the probes of
the held-out measurement, which counted such answers as misses.

The comparison now drops a comma with a digit before it and exactly three after.
The second kind (a reference that carries a unit word the answer omits) is the
generator's reference shape and is not repaired here.

How to avoid it: when a verified-pass rate looks low on facts a stronger model
plainly read, read the failed verdicts before training; and re-grade any
earlier measurement made with a verifier that has since been repaired.
