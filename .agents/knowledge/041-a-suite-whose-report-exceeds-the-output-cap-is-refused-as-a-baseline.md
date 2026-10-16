# 041. A suite whose per-test report exceeds the output cap is refused as a baseline

The first repository families had 575 and 618 tests. `practice` stopped in 1.5 s
with `the baseline is refused (OutputTruncated, 1)`: the generator reads the
runner's per-test report (`python3 -m unittest -v`, about 110 bytes a test)
from a stream the sandbox cuts at 64 KiB, and the two reports were 69795 and
71311 bytes. A third family at 56286 bytes qualified but would have lost
almost every fault: a fault that fails tests adds a traceback per test to the
same stream, and a candidate whose report is cut is refused as
`output_truncated`. Even a 352-test family (40 KB) lost 2 of 20 candidates
that way.

How to avoid it: size a family's passing verbose report to about 30000 bytes
(250 to 300 tests, short test class names) and check it before a family is
admitted: `scripts/tasks/families.py check` fails above 32000 bytes. The cap
is a property of the generator's command runs, not of the project, so a large
real repository is refused the same way; reading the report from a file
instead of the stream would lift it.
