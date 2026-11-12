# 108. One variable typed two ways collapses the task yield

Packs from the per-kind passes generated 2 tasks on the held-out corpus (60 on the fixture) because one variable, `deposit_months`, was declared `int` in
one object and `number` in another and compiling the ruleset raised on it: the whole document group produced nothing. Admission was fine; only the yield
showed it. Make a variable one type per name across the proposals (`int` and `number` become `number`; `unify_variable_types`), widen the same way when
compiling, and watch tasks per document, not only admitted objects.
