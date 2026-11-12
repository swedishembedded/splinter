# 107. Admission guarantees traceability, not usefulness

A model proposing knowledge objects from documents can have every proposal admitted (quote verbatim, every number in the quote) and still miss half of
what a careful reader extracts: the first real run through `splinter objects propose` admitted 27 of 27 and had recall 0.51 against a gold set, with
placeholder fact keys, no formulas and behaviour read as facts. Measure against a gold set in pack form (`crates/knowledge/tests/fixtures/material/gold.json`,
and `material-heldout`, written before any model ran on it): single pass 0.59 / 0.55 recall (fixture / held-out) at 0.82 / 0.73 precision; one pass per
family of kinds 0.82 / 0.80; plus a worked example from an unrelated domain 1.00 / 0.95 (precision about 0.45 until restatements are dropped); the shipped
default 0.90 / 0.90 at 0.61 / 0.72 (a rerun 0.92 / 0.85, so noise is about 0.05). Splitting documents into 60-word chunks never helped and cost held-out
recall (0.65). A brief tuned on one corpus must be judged on one it was not tuned on.
