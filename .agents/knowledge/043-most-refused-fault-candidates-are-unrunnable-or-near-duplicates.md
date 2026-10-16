# 043. Most fault candidates of a repository are refused, and the refusals are not the tests' fault

Run with 80 candidates per file over three Python families of 257 to 279 tests
(701, 828 and 916 candidates), the generator admitted 159, 177 and 186 tasks.
Of the candidates, 27 to 34 percent did not run (`broken_build`: a line removed
or an operator changed so the module no longer loads, which the suite sees as a
different set of tests), 9 to 13 percent were equivalent (no test noticed),
and 29 to 36 percent were refused as `near_duplicate`: requests are a constant
paragraph plus the names of the failing tests, so two faults that fail the same
tests, or tests with similar names, make the same request. A further 1 to 12
percent were `output_truncated` even at a 28 KB passing report (see 041).

What this means: a family's admitted count is bounded by the number of
distinct failing-test sets, not by its size, so test names must be varied
words (4 to 7, never 8 or more, which a quotation check refuses). The
count of candidates that "the tests catch" is not an admission rate; report
the admitted number and the refusals by reason, as the `[tasks]` line gives them.
