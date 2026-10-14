# 021. A held-out suite that held paraphrases of trained tasks measured recall

The release gate's improvement suite absorbed the variants of tasks that a
trained record was projected from. Most of what it called held-out was a
paraphrase of a trained task, so the candidate won it by a wide, significant
margin, while on the tasks that were truly held out the same candidate showed
nothing. The check measured memory of taught facts under paraphrase against the
base under the default prompt; it said nothing about unseen letters or about
beating the prompted base.

How to avoid it: build a held-out suite from families the training never saw,
keep generalisation to held-out records and recall under paraphrase apart in
the report (each with its own difference, interval and test), and state the
prompt each arm was asked each suite under. Recompute any headline from the
tasks it is made of before believing it; count how many of its units are
held out.
