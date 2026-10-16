<!--
SPDX-License-Identifier: CC-BY-4.0
Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
-->

# Citation check of the correction-plan input, 2026-10-16 (package R0)

Method: each id was looked up on its arXiv abstract page (`arxiv.org/abs/<id>`) on
2026-10-16. Recorded: whether the id exists, the title the page gives, the year and month
of the first submission, and whether title and claimed use match the plan's wording. Only
the abstract page was read, through a summarizing fetch; claims below about content are
what the abstract states, not a reading of the paper. Entries are cited by title, year and
arXiv id only.

Result: 8 of 8 ids exist. 6 match the plan's name and use, 2 are mismatched in name or
topic. Only the 6 verified entries (and the 2 with the stated caveat) may enter a
`references.bib`.

| Plan name | arXiv id | Page title | First submitted | Verdict |
|---|---|---|---|---|
| SWE-smith | 2504.21798 | SWE-smith: Scaling Data for Software Engineering Agents | 2025-04 | verified |
| Agentic RL for real-world code repair | 2510.22075 | Agentic Reinforcement Learning for Real-World Code Repair | 2025-10 | verified (title matches; claimed use "transfer across environments is hard" not read beyond the title) |
| SWE-Bench Pro Verified | 2609.08149 | SWE-Bench Pro Verified: A Reliable Benchmark for Software Engineering Agents | 2026-09 (revised to v3 on 2026-10-09) | verified; the page is revised, so a citation must name the version read |
| Harness-Bench | 2605.27922 | Harness-Bench: Evaluating Agent Harnesses across Models and Realistic Workflows | 2026-05 | verified |
| HARBOR | 2604.20938 | HARBOR: Automated Harness Optimization | 2026-04 | verified; the abstract frames harness tuning as constrained noisy Bayesian optimization with a multi-fidelity cost-aware acquisition, which matches "constrained cost-aware harness search" |
| Mixture of Self-Improving Branches | 2609.37834 | Mixture of Self-Improving Branches For Agent Harness Optimization | 2026-09 | verified; the suspicion in the plan (id suffix above five digits) was wrong; the abstract discusses complementary harnesses from branches and a router |
| forgetting mitigation | 2505.17496 | Analyzing Mitigation Strategies for Catastrophic Forgetting in End-to-End Training of Spoken Language Models | 2025-05 | mismatched scope: exists and its abstract names experience replay as the best of three strategies, but it studies spoken language models, not agentic or code models; usable only as evidence that replay helps in that setting, and the plan's name "forgetting mitigation" is not a title |
| Online Process Reward Learning | 2509.19199 | Agentic Reinforcement Learning with Implicit Step Rewards | 2025-09 | mismatched name: the id exists and is about step-level credit assignment for agent RL, but the abstract does not use the name "Online Process Reward Learning" (it names its method differently); cite by the page title, never by the plan's name |

## Consequences for the plan

* No id was fabricated. The post-dated and five-digit-suffix suspicions in the master plan
  section 1.1 are resolved: 2609.08149 and 2609.37834 exist.
* Two names in the plan are not paper titles. Any text, bib entry or code comment that
  uses "forgetting mitigation" or "Online Process Reward Learning" must be replaced by
  the page titles above.
* The claimed uses are the plan's, not the papers'. This check supports existence and
  title only; each claim that a paper shows a given effect needs a read of the paper
  before it enters evidence.
