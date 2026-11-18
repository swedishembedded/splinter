<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 174. An examiner's signature verifies only over its own float spelling

**Lesson:** verifying bench's HMAC in-process means reproducing bench's canonical bytes exactly,
and "sorted compact JSON" is not enough: bench's serialiser and Rust's spell the same float
differently. Test the verifier against a document bench itself signed, not only against documents
the verifier's own code signed.

Found 2026-11-18 while moving signed-verdict verification into the promotion gate
(`splinter_eval::promotion`). Bench signs `canonical_bytes(document)` (sorted keys, compact,
UTF-8). Its serialiser writes shortest round-trip digits positionally for a decimal exponent from
-5 to 15 and scientifically with a signed exponent otherwise: `2.5e-5` is `0.000025`, `1e16` is
`1e+16`, a `p_one_sided` of 2^-16 is `0.0000152587890625`. `serde_json` writes `2.5e-5`, `1e16`
and `1.52587890625e-5`, so a MAC over `serde_json` bytes fails on real verdicts that carry such
values while passing every self-signed test. `promotion::canonical` implements bench's rule; its
table of spellings was printed by bench's own serialiser, and
`promotion_gate::a_verdict_the_examiner_signed_verifies_here_and_an_altered_or_forged_one_does_not`
verifies a fixture `bench examine` signed (`crates/eval/tests/fixtures/`).
