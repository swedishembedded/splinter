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
# The build directory `make local` configures. Sharing sven's lets brain's
# and sven's compiled crates be reused rather than built a second time.
TARGET_DIR ?= $(SVEN_DIR)/target

CARGO ?= cargo
# Release everywhere, tests included: brain's kernels are unusably slow
# unoptimised, and one profile means one set of compiled dependencies.
PROFILE := --release

.PHONY: help local lock build test fmt check check/gates check/fmt check/clippy \
	hooks/install experiments/tool-syntax/audit

## help - list the targets
help:
	@grep -E '^## ' $(MAKEFILE_LIST) | sed 's/^## /  /'

## local - build against local sven/brain checkouts (SVEN_DIR, BRAIN_DIR, TARGET_DIR); writes the gitignored .cargo/config.toml
local:
	bash scripts/local-config.sh $(SVEN_DIR) $(BRAIN_DIR) $(TARGET_DIR)

## lock - pin Cargo.lock to the HEADs of SVEN_DIR and BRAIN_DIR (push those commits before sharing the lock)
lock:
	bash scripts/pin-lock.sh $(SVEN_DIR) $(BRAIN_DIR)

## build - build every package
build:
	$(CARGO) build $(PROFILE) --workspace

## test - run every test (no model, no GPU needed)
test:
	$(CARGO) test $(PROFILE) --workspace

## fmt - format Splinter's own sources (never a dependency's)
fmt:
	rustfmt --edition 2021 $$(git ls-files --cached --others --exclude-standard '*.rs')

## check - every gate over the whole tree (what the hooks check per commit)
check: check/gates check/fmt check/clippy

## check/gates - text hygiene, headers, paths, file sizes, citations, manifests, lock sources, repo scope, history
check/gates:
	python3 scripts/hooks/text-hygiene.py
	python3 scripts/spdx/check.py $$(git ls-files)
	bash scripts/gates/check-no-machine-paths.sh
	bash scripts/gates/check-large-files.sh
	bash scripts/gates/check-no-doc-citations.sh
	bash scripts/gates/check-manifests.sh
	bash scripts/gates/check-lock-sources.sh
	bash scripts/gates/check-repo-scope.sh
	bash scripts/gates/check-linear-history.sh

## check/fmt - formatting of Splinter's own sources
check/fmt:
	rustfmt --edition 2021 --check $$(git ls-files --cached --others --exclude-standard '*.rs')

## check/clippy - clippy on every target, warnings denied
check/clippy:
	$(CARGO) clippy $(PROFILE) --workspace --all-targets -- -D warnings

## hooks/install - install the pre-commit, commit-msg and pre-push hooks (once per clone)
hooks/install:
	pre-commit install --install-hooks

## experiments/tool-syntax/audit - the task catalog checks itself (no model, no GPU, no network)
experiments/tool-syntax/audit:
	$(CARGO) run $(PROFILE) -p splinter-tool-syntax -- audit
