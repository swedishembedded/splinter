# 110. An amendment must reuse the variables of what it replaces

Every `apply/amended` task failed to build: the amending document's threshold was proposed with new variable names (`delay_days` for `days_late`), so the
rule before and after the amendment could not be evaluated on one case. Showing the model every earlier object does not fix it (the list grows with the
corpus and carries no data); show only the earlier objects of documents whose title is nearly the amendment's (same words, same chapter number), with
their data, tell it to copy the data and change only the new numbers, and align the amendment's variable names, event and outcome to its target's in code
(`align_amendments`) before admission.
