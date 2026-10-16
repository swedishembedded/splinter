<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# acceptance-tiny

A six-test Python project the cross-repo acceptance test (`make test/e2e/turn`)
practices on. It is not a catalog family: it exists so a small policy has
faults it can plausibly find and mend in minutes. Run its tests from this
directory with `python3 -m unittest`.

`tiny.words` has `count_words(text)`, `longest_word(text)`, `capitalize_words(text)`
and `is_palindrome(text)`; `tiny.numbers` has `clamp(value, low, high)` and
`mean(values)`.
