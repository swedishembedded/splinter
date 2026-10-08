# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements self-improving agent systems for its
# clients. If your team needs expertise in continual learning or agent
# evaluation, you can procure our services by sending an email to
# info@swedishembedded.com.

SHELL := bash

# Local checkouts, for `make local` and `make lock` only. A plain build uses
# the remotes pinned in Cargo.lock.
SVEN_DIR ?= ../sven
BRAIN_DIR ?= ../edgeai/brain

CARGO ?= cargo
# Release everywhere, tests included: brain's kernels are unusably slow
# unoptimised, and one profile means one set of compiled dependencies.
PROFILE := --release

# Splinter's own Rust sources: tracked or new, and present in the working
# tree (a tracked file deleted there is no longer a source file).
RUST_SOURCES = $(wildcard $(shell git ls-files --cached --others --exclude-standard '*.rs'))

.PHONY: help local lock build test fmt check check/gates check/fmt check/clippy
	hooks/install samples/tool-syntax/audit \

## help - list the targets
help:
	@grep -E '^## ' $(MAKEFILE_LIST) | sed 's/^## /  /'

## local - build against local sven/brain checkouts (SVEN_DIR, BRAIN_DIR); writes the gitignored .cargo/config.toml
local:
	bash scripts/local-config.sh $(SVEN_DIR) $(BRAIN_DIR)

## lock - pin Cargo.lock to the HEADs of SVEN_DIR and BRAIN_DIR (push those commits before sharing the lock)
lock:
	bash scripts/pin-lock.sh $(SVEN_DIR) $(BRAIN_DIR)

## build - build every package
build:
	$(CARGO) build $(PROFILE) --workspace

## test - run every test (no model, no GPU needed)
test:
	$(CARGO) test $(PROFILE) --workspace
	python3 samples/adams/test_fetch.py

## papers/<name>/pdf - build one paper's PDF (see papers/paper.mk); papers/pdf builds every paper
PAPER_DIRS := $(patsubst %/Makefile,%,$(wildcard papers/*/Makefile))
PAPER_TARGETS := $(addsuffix /pdf,$(PAPER_DIRS))

papers/pdf: $(PAPER_TARGETS)

.PHONY: papers/pdf $(PAPER_TARGETS)

$(PAPER_TARGETS): papers/%/pdf:
	$(MAKE) -C papers/$*

## fmt - format Splinter's own sources (never a dependency's)
fmt:
	rustfmt --edition 2021 $(RUST_SOURCES)

## check - every gate over the whole tree (what the hooks check per commit)
check: check/gates check/fmt check/clippy

## check/gates - text hygiene, headers, paths, file sizes, citations, manifests, lock sources,
##               repo scope, environment reads, crate layering, history
check/gates:
	python3 scripts/hooks/text-hygiene.py
	python3 scripts/spdx/check.py $$(git ls-files)
	bash scripts/gates/check-no-machine-paths.sh
	bash scripts/gates/check-large-files.sh
	bash scripts/gates/check-no-doc-citations.sh
	bash scripts/gates/check-manifests.sh
	bash scripts/gates/check-lock-sources.sh
	bash scripts/gates/check-repo-scope.sh
	bash scripts/gates/check-env-reads.sh
	bash scripts/gates/check-architecture.sh
	bash scripts/gates/check-scripts.sh
	bash scripts/gates/check-no-perf-numbers.sh
	bash scripts/gates/check-linear-history.sh

## check/fmt - formatting of Splinter's own sources
check/fmt:
	rustfmt --edition 2021 --check $(RUST_SOURCES)

## check/clippy - clippy on every target, warnings denied
check/clippy:
	$(CARGO) clippy $(PROFILE) --workspace --all-targets -- -D warnings

## hooks/install - install the pre-commit, commit-msg and pre-push hooks (once per clone)
hooks/install:
	pre-commit install --install-hooks

## samples/tool-syntax/audit - the task catalog checks itself (no model, no GPU, no network)
samples/tool-syntax/audit:
	$(CARGO) run $(PROFILE) -p splinter-tool-syntax -- audit
