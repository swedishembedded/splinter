# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements self-improving agent systems for its
# clients. If your team needs expertise in continual learning or agent
# evaluation, you can procure our services by sending an email to
# info@swedishembedded.com.

SHELL := bash

# Local checkouts, for `make deps/local` and `make deps/lock` only. A plain build uses
# the remotes pinned in Cargo.lock.
SVEN_DIR ?= ../sven
BRAIN_DIR ?= ../edgeai/brain

# The binaries and model store `make test/e2e/...` runs against.
BRAIN_BIN ?= $(BRAIN_DIR)/target/release/brain
SVEN_BIN ?= $(SVEN_DIR)/target/release/sven
SPLINTER_BIN ?= target/release/splinter
MODELS_DIR ?= $(HOME)/.local/share/brain/models

CARGO ?= cargo

# Shared-node lease kit (scripts/lease). One root holds the lane worktrees, the
# per-lane cargo targets and the lock files.
LEASE_ROOT ?= $(HOME)/project/shared/containers
LEASE_REPO ?= .
LEASE_LANE ?=
GPU_LOCK ?= $(LEASE_ROOT)/gpu.lock
# Release everywhere, tests included: brain's kernels are unusably slow
# unoptimised, and one profile means one set of compiled dependencies.
PROFILE := --release

# Splinter's own Rust sources: tracked or new, and present in the working
# tree (a tracked file deleted there is no longer a source file).
RUST_SOURCES = $(wildcard $(shell git ls-files --cached --others --exclude-standard '*.rs'))

# Every workspace package (crates and samples): one sed over the manifests, so
# adding a crate adds its build/crate/<package> and test/crate/<package>
# targets. The first `name =` after [package] is the package name.
CRATE_PACKAGES := $(shell sed -sn '/^\[package\]/,/^name *=/s/^name *= *"\(.*\)"/\1/p' crates/*/Cargo.toml samples/*/Cargo.toml samples/*/*/Cargo.toml)

# MAKE TARGETS (rule: AGENTS.md, "Make targets"). A target is
# `make <concept>/<subconcept>/.../<action>`: concept first, verb last, one
# segment per directory level, and every prefix is itself a target so
# `make <TAB>` descends one level at a time. Where a set follows from the tree
# (gates, crates, papers) it is GENERATED as explicit targets by static pattern
# rules over a computed list - never `%` pattern rules, which bash completion
# cannot see - so adding a script, crate or paper adds a target. `make help`
# prints the tree from the ##@ section and ## doc lines;
# scripts/gates/check-make-targets.sh (check/gates/make-targets) enforces the
# naming. An area target is either the default action for its area (build,
# test, check, papers) or, where no default is safe or sensible (deps, hooks,
# samples, build/crate, test/crate), only lists its children and does nothing
# else.

# ---------------------------------------------------------------------------
##@ Build and test
## The whole workspace by default; one package on demand.

## build - default: build every package (release profile)
build:
	$(CARGO) build $(PROFILE) --workspace

## build/crate - lists the per-package build targets; builds nothing
build/crate:
	@bash scripts/make-help.sh build/crate

## build/crate/<package> - build one workspace package (not part of `make build`)
BUILD_CRATE_TARGETS := $(addprefix build/crate/,$(CRATE_PACKAGES))
$(BUILD_CRATE_TARGETS): build/crate/%:
	$(CARGO) build $(PROFILE) -p $*

## test - default: run every test (no model, no GPU needed)
test:
	$(CARGO) test $(PROFILE) --workspace
	python3 samples/adams/test_fetch.py

## test/crate - lists the per-package test targets; tests nothing
test/crate:
	@bash scripts/make-help.sh test/crate

## test/crate/<package> - run one workspace package's tests (not part of `make test`)
TEST_CRATE_TARGETS := $(addprefix test/crate/,$(CRATE_PACKAGES))
$(TEST_CRATE_TARGETS): test/crate/%:
	$(CARGO) test $(PROFILE) -p $*

## test/families - the repository task families: baselines pass identically five times, conventions hold, faults are caught (no model, no GPU, minutes)
test/families:
	python3 scripts/tasks/families.py check
	python3 scripts/tasks/families.py mutate
	python3 scripts/tasks/test_runaway.py

## test/e2e - lists the end-to-end targets, which need built brain and sven binaries; runs nothing
test/e2e:
	@bash scripts/make-help.sh test/e2e

## test/e2e/brain-sven - brain serves a model and sven completes a tool-using task with it (BRAIN_BIN, SVEN_BIN, MODELS_DIR; CPU backend, no GPU, minutes to tens of minutes; exit 3 means a prerequisite is missing and nothing was tested)
test/e2e/brain-sven:
	bash scripts/e2e/brain-sven.sh $(BRAIN_BIN) $(SVEN_BIN) $(MODELS_DIR)

## test/e2e/turn - one full turn on Qwen3-0.6B: practice on a tiny fixture, experiences with verdicts, a trained candidate, the release gate, the candidate served and driven by sven (SPLINTER_BIN, BRAIN_BIN, SVEN_BIN, MODELS_DIR; DEVICE in brain's grammar, e.g. DEVICE=cpu for the CPU backend; for the short GPU variant run it under `make lease/gpu/run`; exit 3 means a prerequisite is missing and nothing was tested)
test/e2e/turn:
	bash scripts/e2e/turn.sh $(SPLINTER_BIN) $(BRAIN_BIN) $(SVEN_BIN) $(MODELS_DIR) tasks/acceptance-tiny $(DEVICE)

## test/e2e/turn/spec - the spec of test/e2e/turn against a fake splinter, brain and sven: each missing artifact fails the run with its name
test/e2e/turn/spec:
	python3 -I scripts/e2e/test-turn.py

# ---------------------------------------------------------------------------
##@ Checks
## Every gate over the whole tree: what the hooks check per commit.

## check - default: check/gates, check/fmt, check/clippy, check/papers
check: check/gates check/fmt check/clippy check/papers

# Text gates: scripts/gates/check-<gate>.sh is the target check/gates/<gate>.
# The two Python gates are written out. A gate that needs flags would be an
# explicit override of its generated rule.
GATE_NAMES := $(patsubst scripts/gates/check-%.sh,%,$(wildcard scripts/gates/check-*.sh))
GATE_TARGETS := $(addprefix check/gates/,$(GATE_NAMES))

## check/gates - default: every text gate below (hygiene, headers, paths, file sizes,
##   citations, manifests, lock sources, repo scope, environment reads, seed literals,
##   crate layering, history, make target naming)
check/gates: check/gates/text-hygiene check/gates/spdx $(GATE_TARGETS)

## check/gates/text-hygiene - ASCII punctuation, whitespace, line endings, parseable files
check/gates/text-hygiene:
	python3 scripts/hooks/text-hygiene.py

## check/gates/spdx - every tracked file carries its SPDX and copyright header
check/gates/spdx:
	python3 scripts/spdx/check.py $$(git ls-files)

## check/gates/<gate> - run scripts/gates/check-<gate>.sh alone (one target per script)
##   Each script's header says what it enforces.
$(GATE_TARGETS): check/gates/%: scripts/gates/check-%.sh
	bash $<

## check/papers - every paper's claims ledger re-verified at its pinned commits (needs sibling sven and brain checkouts)
check/papers:
	$(MAKE) -C papers/splinter-brain-sven-flywheel verify

## check/fmt - formatting of Splinter's own sources
check/fmt:
	rustfmt --edition 2021 --check $(RUST_SOURCES)

## check/clippy - clippy on every target, warnings denied
check/clippy:
	$(CARGO) clippy $(PROFILE) --workspace --all-targets -- -D warnings

# ---------------------------------------------------------------------------
##@ Papers and samples
## Paper PDFs, and the sample applications' own actions.

PAPER_DIRS := $(patsubst %/Makefile,%,$(wildcard papers/*/Makefile))
PAPER_TARGETS := $(addsuffix /pdf,$(PAPER_DIRS))

## papers - default: papers/pdf (every paper's PDF; see papers/paper.mk)
papers: papers/pdf

## papers/pdf - build every paper's PDF
papers/pdf: $(PAPER_TARGETS)

## papers/<name> - build that paper's PDF (same as papers/<name>/pdf)
$(PAPER_DIRS): %: %/pdf

## papers/<name>/pdf - build one paper's PDF
$(PAPER_TARGETS): papers/%/pdf:
	$(MAKE) -C papers/$*

## samples - lists the sample targets; runs nothing (build one with build/crate/<package>)
samples:
	@bash scripts/make-help.sh samples

## samples/generations - lists the generations sample's targets; runs nothing
samples/generations:
	@bash scripts/make-help.sh samples/generations

## samples/generations/anchor-200 - build the 200-question anchor suite into target/samples/generations/ (generated, gitignored)
samples/generations/anchor-200:
	python3 samples/generations/anchor200.py target/samples/generations/anchor-200.jsonl

## samples/tool-syntax - lists the tool-syntax sample's targets; runs nothing
samples/tool-syntax:
	@bash scripts/make-help.sh samples/tool-syntax

## samples/tool-syntax/audit - the task catalog checks itself (no model, no GPU, no network)
samples/tool-syntax/audit:
	$(CARGO) run $(PROFILE) -p splinter-tool-syntax -- audit

# ---------------------------------------------------------------------------
##@ Lease kit
## Wrappers for the shared leased node: build slots, the GPU lock, lane worktrees.

## lease - run the kit's specification, then print the node report
lease: lease/test lease/doctor

## lease/test - the lease kit's specification (no GPU, no network)
lease/test:
	python3 -I scripts/lease/test-lease.py

## lease/doctor - node, lease end, memory cap, disk, GPU contexts, lock holder, worktrees, target sizes
lease/doctor:
	bash scripts/lease/doctor --repo $(LEASE_REPO) --lock $(GPU_LOCK) --disk $(LEASE_ROOT) \
		--worktrees $(LEASE_ROOT)/worktrees --targets $(LEASE_ROOT)/targets

## lease/gpu - lists the GPU lock targets; does nothing
lease/gpu:
	@bash scripts/make-help.sh lease/gpu

## lease/gpu/run - run CMD under the GPU lock and record foreign contexts (EXCLUSIVE=1 refuses a busy GPU)
lease/gpu/run:
	bash scripts/lease/gpu --lock $(GPU_LOCK) $(if $(EXCLUSIVE),--exclusive) -- $(CMD)

## lease/card - lists the per-card GPU reservation targets; does nothing
lease/card:
	@bash scripts/make-help.sh lease/card

## lease/card/run - run CMD with a GB GiB VRAM reservation on one GPU (CARD=0|1|any, default any; blocks until the shares are free; the command sees CARD and BRAIN_DEVICE_FLAG)
lease/card/run:
	bash scripts/lease/card --lock-dir $(LEASE_ROOT)/locks --card $(or $(CARD),any) --gb $(GB) --wait -- $(CMD)

## lease/worktree - lists the lane worktree targets; does nothing
lease/worktree:
	@bash scripts/make-help.sh lease/worktree

## lease/worktree/add - create LEASE_LANE's worktree (branch <lane>-<date> from origin/main) and target
lease/worktree/add:
	bash scripts/lease/worktree add $(LEASE_LANE) --repo $(LEASE_REPO) \
		--worktrees $(LEASE_ROOT)/worktrees --targets $(LEASE_ROOT)/targets

## lease/worktree/list - lane worktrees of LEASE_REPO and the size of each lane target
lease/worktree/list:
	bash scripts/lease/worktree list --repo $(LEASE_REPO) \
		--worktrees $(LEASE_ROOT)/worktrees --targets $(LEASE_ROOT)/targets

## lease/worktree/remove - delete LEASE_LANE's worktree and target (the branch is kept)
lease/worktree/remove:
	bash scripts/lease/worktree remove $(LEASE_LANE) --repo $(LEASE_REPO) \
		--worktrees $(LEASE_ROOT)/worktrees --targets $(LEASE_ROOT)/targets

# ---------------------------------------------------------------------------
##@ Dependencies and tooling
## Local sven/brain checkouts, git hooks, formatting.

## deps - lists the dependency targets; does nothing
deps:
	@bash scripts/make-help.sh deps

## deps/local - build against local sven/brain checkouts (SVEN_DIR, BRAIN_DIR); writes the gitignored .cargo/config.toml
deps/local:
	bash scripts/local-config.sh $(SVEN_DIR) $(BRAIN_DIR)

## deps/lock - pin Cargo.lock to the HEADs of SVEN_DIR and BRAIN_DIR (push those commits before sharing the lock)
deps/lock:
	bash scripts/pin-lock.sh $(SVEN_DIR) $(BRAIN_DIR)

## hooks - lists the hook targets; installs nothing
hooks:
	@bash scripts/make-help.sh hooks

## hooks/install - install the pre-commit, commit-msg and pre-push hooks (once per clone)
hooks/install:
	pre-commit install --install-hooks

## fmt - format Splinter's own sources, never a dependency's (the bare leaf: one action; `check/fmt` is its checking twin)
fmt:
	rustfmt --edition 2021 $(RUST_SOURCES)

## help - show every target as a tree, grouped by section
help:
	@bash scripts/make-help.sh

# Declared last so every generated list above is already defined.
.PHONY: help build build/crate $(BUILD_CRATE_TARGETS) test test/crate $(TEST_CRATE_TARGETS) test/e2e test/e2e/brain-sven test/e2e/turn test/e2e/turn/spec test/families fmt \
	check check/fmt check/clippy check/papers check/gates check/gates/text-hygiene check/gates/spdx $(GATE_TARGETS) \
	lease lease/test lease/doctor lease/gpu lease/gpu/run lease/card lease/card/run lease/worktree lease/worktree/add lease/worktree/list lease/worktree/remove \
	deps deps/local deps/lock \
	hooks hooks/install \
	papers papers/pdf $(PAPER_DIRS) $(PAPER_TARGETS) \
	samples samples/generations samples/generations/anchor-200 samples/tool-syntax samples/tool-syntax/audit
