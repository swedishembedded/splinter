<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Make target rules

Read this before you add, rename or document a Makefile target.
`scripts/gates/check-make-targets.sh` (`make check/gates/make-targets`, also a
pre-commit hook) enforces the machine-checkable part, and `make help` prints
the tree from the same doc lines through `scripts/make-help.sh`.

Every public make target is `make <concept>/<subconcept>/.../<action>`:

- The concept comes first (a noun: `build`, `test`, `check`, `deps`, `hooks`,
  `papers`, `samples`), the verb or variant comes last, and there are as many
  levels as the thing needs, usually two or three. A test or sample hierarchy
  with subdirectories gets one segment per directory level:
  `samples/tool-syntax/audit`. Never put the verb or area second.
- Names are lowercase `[a-z0-9-]` segments joined by `/`. Private helper
  targets start with `_`.
- Every prefix is itself a target, so `make <TAB>` descends one level at a
  time. An area target is the default action for its area (`build`, `test`,
  `check`, `papers`, `docs`: it runs its default set, named in its doc line)
  or, where no default is safe or sensible (`deps`, `hooks`, `samples`,
  `build/crate`, `test/crate`), it only lists its children and does nothing
  else. An umbrella of anything that tags, pushes or uploads never acts.
- A set that follows from the tree is generated, never hand-listed: explicit
  targets from a static pattern rule over a computed list (never `%` pattern
  rules, which bash completion cannot see). Here: `check/gates/<gate>` (one per
  `scripts/gates/check-<gate>.sh`), `test/crate/<package>`,
  `build/crate/<package>`, `papers/<name>/pdf`. Adding a file or directory
  adds its target.
- A bare single-segment target is allowed only for an area, or for the leaves
  `fmt` and `help`.
- Every target has a `## name - description` line (a generated family has one
  line with a `<placeholder>`) and sits under a `##@ Section` line, and is
  listed in `.PHONY`.
