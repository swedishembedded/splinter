# 028. A drawn seed made three tests compare different things

Once a missing seed is drawn and recorded (instead of a fixed default), a test
that leaves the seed out stops being deterministic, and it fails in a way that
looks like a product defect:

- `expdb` selection: two compilations of one recipe were asserted to have one
  plan id. The plan records the drawn seed, so the ids differed on every run
  (3 of 3 runs). The recipe now fixes its seed, because "a fixed recipe
  compiles to one plan" is the property under test.
- `pipelines` variants: five tests failed at once with "the denoise kind needs
  the seed it is built with" (5 of 6 tests in the file).
- `pipelines` release_workspace: the exam families came from a drawn seed, and
  on some draws the fixture left one passing training record where training
  needs two ("1 record(s) passed; training holds records out for scoring and
  needs at least 2"). 2 of 5 runs failed, a different test each time; with the
  seed fixed, 5 of 5 whole-file runs passed.

How to avoid it: a test that depends on a fixture's shape fixes the seed and
says why with a `seed:` comment; a test about determinism fixes the seed too.
A flaky test that fails on a different test each run is a seed, not a race,
until the report shows otherwise.
