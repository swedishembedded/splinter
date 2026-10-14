# 020. An applied patch was the redacted copy, not the candidate

The coding-loop sample redacts credential-shaped text from everything it
persists, and it passed the patch files through the same filter. `apply` then
applied the file it had written: a candidate that added `token = next_token()`
became `token = [REDACTED]` in the patch, so the repository received code the
loop had never judged, or the patch no longer applied because a context line
had been rewritten. The filter matches names that sound like secrets (token,
password, secret, key) followed by an assignment, which is ordinary source text
in most repositories.

The patch files are now written exactly as the candidate was read back from
git, and only the trace and the tails of check output are redacted. The spec
`the_patch_that_is_applied_is_the_candidate_exactly_even_where_its_text_looks_like_a_secret`
fixes the seeded bug with a function that assigns `token = 5`, applies the run's
patch and compares the file byte for byte: the filtered patch writes
`token = [REDACTED]` there.

How to avoid it: a redaction filter belongs on records that are read, never on
an artifact that is executed or applied. If the artifact itself may hold a
secret, refuse to keep it; do not keep a different one under its name.
