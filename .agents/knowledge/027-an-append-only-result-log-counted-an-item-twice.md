# 027. An append-only result log counted an item twice

Two exam runs of one arm overlapped and appended to the same log, so 81 items
had two lines and the report counted 241 items instead of 160. Resuming was
safe (an item already in the log is not asked again); reading was not. The log
is the truth about what finished, and a reader that does not key its records
believes a record written twice.

The shared job log now reads each key once, the first record standing
(store::jsonl::read_unique), and appends under a lock so that records written
from many threads never share a line. How to avoid it: key every read of an
append-only log, and run one writer per log or accept that the reader dedupes.
