<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 079. A test that searched a whole dataset file for "8080" failed when a blake3 hash contained it

`absorb` test `a_night_teaches_every_claim_from_many_records_and_releases_what_it_answers` asserted that the
dataset text held neither the agent's wrong port ("8080") nor its capitulation. It searched the raw JSON lines, whose
`metadata` carries blake3 hex (experience, task and group ids). The hex of one record happened to contain the
substring `98080972`, so the test failed on origin/main although nothing in the data was wrong; which hashes
appear depends on every hashed input, so any unrelated commit that changes a record's digest can flip the test either way
(it passed on one revision and failed on the next with no change to the absorb code).

The assertion now reads only the `content` of each record's messages, which is what "never trained on" is about. A
negative substring check over a serialized file that also holds hashes needs a needle that cannot occur in hex.
