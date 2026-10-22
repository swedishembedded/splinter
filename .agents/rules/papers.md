<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Research paper rules

Read this before you write or change anything under `papers/`, or make a
change that produces or invalidates a finding a paper states.

`papers/` holds research papers, not plans. A paper is a proper paper in the
style of an arXiv submission (abstract, method, evaluation, results,
limitations, references) that builds to PDF. A final paper contains only
findings that are proven and verified. While a paper is a draft (a paper with a
claims ledger records it in `status.txt`) it may also hold registered work whose
results are not yet validated, stated as such with empty result tables; a
paper's `final` target, where it has one, fails while any registered or planned
claim remains. Roadmaps, intentions, open questions and next steps belong in
`.agents/roadmap/` and `.agents/research/`; a paper that reads like one of them
is wrong.

- **Route by scope.** Every paper states its scope in its introduction, and
  `papers/README.md` lists the scopes. Put a new finding in the paper whose
  scope it falls in; if none fits, start a new paper directory and add its
  scope there.
- **Keep papers current.** A change that produces a new verified finding, or
  invalidates a stated one (a corrected defect, a re-measured result, a changed
  method), updates the affected paper in the same change: text, tables,
  figures and the numbers they rest on.
- **Verified means reproducible.** Every number traces to a measurement that
  can be rerun with the commands the paper documents, comes with its sample
  size and uncertainty, and is labelled as a pilot when the evaluation was
  underpowered or not frozen in advance. A result that was not measured is
  absent, never zero; a result that did not hold is reported as such.
- **No speculation as findings.** Claims the evidence does not support stay out
  or are stated as limitations. Every citation is checked to exist.
- **The paper must build.** `make papers/<name>/pdf` (or `make` in the paper
  directory) produces the PDF with no undefined references or citations before
  the change is committed. The PDF is `target/papers/<name>.pdf` and the LaTeX
  intermediates are under `target/papers/.build/`; nothing is written into the
  paper's directory (`papers/paper.mk`).
- **A claims ledger is re-verified.** `make check/papers` re-checks every
  ledgered claim at its pinned commits; it needs sibling sven and brain
  checkouts.
