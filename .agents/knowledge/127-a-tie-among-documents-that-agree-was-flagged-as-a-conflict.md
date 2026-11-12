<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 127. A tie among documents that agree was flagged as a conflict by the compiler's resolution

`objects::resolve` took the two best-ranked objects of a fact key and flagged a conflict when they had equal rank, without checking that they said different things. A section and an exact copy of it (same authority, date, version, same value) therefore made the whole key conflicted and the duplicate-section test returned `Contested` instead of `Duplicate`. Fix: the strongest competitor is the best object whose content differs from the top one's. Evidence: splinter 2ccdf016d, test `a_learned_section_is_not_chosen_again_and_its_duplicate_is_refused`.
