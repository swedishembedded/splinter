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

# MAKE TARGETS (rule: .agents/rules/make-targets.md). A target is
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
	python3 scripts/qualify/test_qualify.py

## test/crate - lists the per-package test targets; tests nothing
test/crate:
	@bash scripts/make-help.sh test/crate

## test/crate/<package> - run one workspace package's tests (not part of `make test`)
TEST_CRATE_TARGETS := $(addprefix test/crate/,$(CRATE_PACKAGES))
$(TEST_CRATE_TARGETS): test/crate/%:
	$(CARGO) test $(PROFILE) -p $*

## test/package - the spec of the release publishing script against a scratch repository and a fake gh (bats; no build, seconds)
test/package:
	bats scripts/package/publish-deb.bats

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
##   crate layering, history, make target naming, roadmap index)
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

## samples/repo-replay - lists the repo-replay sample's targets; runs nothing
samples/repo-replay:
	@bash scripts/make-help.sh samples/repo-replay

## samples/repo-replay/audit - the gate checks itself on a fixture history with scripted models (no model, no GPU, no network)
samples/repo-replay/audit:
	$(CARGO) run $(PROFILE) -p splinter-repo-replay -- audit

## samples/model-check - lists the model-check sample's targets; runs nothing
samples/model-check:
	@bash scripts/make-help.sh samples/model-check

## samples/model-check/audit - every TLA+ specification, Lua reference implementation and mutation checks itself; no model, no GPU, no network (LUA=<Lua 5.1 interpreter>, TLAPLUS_DIR=<make deps/tlaplus directory>)
samples/model-check/audit:
	test -n "$(LUA)" -a -n "$(TLAPLUS_DIR)" || { echo "samples/model-check/audit: give LUA=<interpreter> and TLAPLUS_DIR=<directory>"; exit 2; }
	$(CARGO) run $(PROFILE) -p splinter-model-check -- audit --lua "$(LUA)" --tlaplus-dir "$(TLAPLUS_DIR)"

## samples/model-check/report - the mutation-detection table of every family: what the explorer and what TLC find in each mutant; no model, no GPU, no network (LUA=, TLAPLUS_DIR= as for the audit)
samples/model-check/report:
	test -n "$(LUA)" -a -n "$(TLAPLUS_DIR)" || { echo "samples/model-check/report: give LUA=<interpreter> and TLAPLUS_DIR=<directory>"; exit 2; }
	$(CARGO) run $(PROFILE) -p splinter-model-check -- report --lua "$(LUA)" --tlaplus-dir "$(TLAPLUS_DIR)"

# ---------------------------------------------------------------------------
##@ Documentation
## The manual under docs/, published as Markdown pages and a PDF by the pinned publisher.
## The contributor notes under .agents/ are not published; their roadmap index is generated.

NODE ?= node
DOCS_PUBLISHER ?= tools/docs-publisher/docs-publisher.mjs
DOCS_OUT := build/docs
DOCS_PDF := build/splinter-manual.pdf
# Every page whose commands the command-line grammar must accept: the manual, the
# READMEs it includes, and the READMEs that introduce the repository and the command.
DOCS_COMMAND_PAGES = $(shell find docs -name '*.md' | sort) README.md crates/splinter/README.md \
	$(shell sed -n 's/^ *- source: //p' docs/manifest.yaml)

# The pin is verified unless DOCS_PUBLISHER is overridden on the command line.
_docs-publisher-pin:
	@$(if $(filter command line,$(origin DOCS_PUBLISHER)),true,cd tools/docs-publisher && sha256sum -c docs-publisher.mjs.sha256 >/dev/null)

## docs - default: docs/build
docs: docs/build

## docs/build - the documentation publication in build/docs (offline: compiles no Rust, loads no model)
docs/build: _docs-publisher-pin
	$(NODE) $(DOCS_PUBLISHER) build --root . --manifest docs/manifest.yaml --out $(DOCS_OUT)

## docs/check - build twice and compare, check links and coverage, then the generated command reference and every command the manual shows against the built SPLINTER_BIN
docs/check: _docs-publisher-pin docs/build
	$(NODE) $(DOCS_PUBLISHER) check --root . --manifest docs/manifest.yaml
	@test -x $(SPLINTER_BIN) || { echo "docs/check: $(SPLINTER_BIN) is missing; build it with: make build/crate/splinter" >&2; exit 1; }
	$(SPLINTER_BIN) cli-reference --check docs/reference/cli.md
	$(SPLINTER_BIN) cli-reference --check-invocations $(DOCS_COMMAND_PAGES)
	python3 scripts/docs/check-site-links.py $(DOCS_OUT) $$( (git ls-files '*.md'; find docs -name '*.md') | grep -v -e '^papers/' -e '^\.agents/' | sort -u)

## docs/pdf - the manual as one PDF in the order of docs/manifest.yaml (needs pandoc and xelatex)
docs/pdf: docs/build
	python3 scripts/docs/pdf.py $(DOCS_OUT) $(DOCS_PDF)

## docs/clean - remove the generated documentation (build/docs and the PDF)
docs/clean:
	rm -rf $(DOCS_OUT) $(DOCS_PDF)

## agents - lists the contributor-notes targets; writes nothing
agents:
	@bash scripts/make-help.sh agents

## agents/roadmap - lists the roadmap targets; writes nothing
agents/roadmap:
	@bash scripts/make-help.sh agents/roadmap

## agents/roadmap/index - write .agents/roadmap/index.md from the ledgers beside it (check/gates/roadmap-index fails on drift)
agents/roadmap/index:
	python3 -I scripts/gates/roadmap-index.py

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
##@ Package and release
## The Debian package of the splinter command line, and the release flow:
## release/{patch,minor,major} bump and tag locally, release/publish is the one
## step that pushes and uploads. package and release only list.

# Read from Cargo.toml on every parse, so the sub-make after a version bump sees
# the new version.
VERSION := $(shell sed -n '/^\[workspace.package\]/,/^\[/s/^version *= *"\(.*\)"/\1/p' Cargo.toml)
# The git remote whose GitHub repository receives the release.
RELEASE_REMOTE ?= origin
RELEASE_DEB = target/debian/splinter_$(VERSION)_$(shell dpkg --print-architecture 2>/dev/null).deb
RELEASE_NOTES = target/release-notes-v$(VERSION).md

## package - lists the package targets; builds nothing
package:
	@bash scripts/make-help.sh package

## package/deb - default: package/deb/release (output in target/debian/)
package/deb: package/deb/release

## package/deb/release - Debian package of the release build of the splinter binary
package/deb/release: build/crate/splinter
	bash scripts/package/build-deb.sh

## package/deb/publish - sign the release package and attach it to the GitHub release of v<version> (created if missing; an earlier upload is replaced); needs gh, a GnuPG key and the tag on GitHub (RELEASE_REMOTE, default origin)
package/deb/publish: release/notes
	bash scripts/package/publish-deb.sh --check --version $(VERSION) --remote $(RELEASE_REMOTE)
	$(MAKE) package/deb/release
	bash scripts/package/publish-deb.sh --version $(VERSION) --remote $(RELEASE_REMOTE) --notes-file $(RELEASE_NOTES) $(RELEASE_DEB)

## release - lists the release targets; does nothing (a release bumps, tags and publishes)
release:
	@bash scripts/make-help.sh release

# Everything a bump needs, checked before any file is touched.
_release/preflight:
	@command -v bump2version >/dev/null 2>&1 || \
		{ echo "error: bump2version not found - install it with 'pip install bump2version'"; exit 1; }
	@[ -z "$$(git status --porcelain --untracked-files=no)" ] || \
		{ echo "error: uncommitted changes (see git status); a release commit holds only the bump"; exit 1; }
	@grep -q '^## \[Unreleased\]$$' CHANGELOG.md || \
		{ echo "error: CHANGELOG.md has no '## [Unreleased]' section to release"; exit 1; }

## release/patch - bump the patch version, date the CHANGELOG section, commit and tag v<version> (local only; release/publish pushes)
release/patch: _release/preflight
	bump2version --no-commit --no-tag patch
	$(MAKE) _release/finalize

## release/minor - bump the minor version, date the CHANGELOG section, commit and tag v<version> (local only)
release/minor: _release/preflight
	bump2version --no-commit --no-tag minor
	$(MAKE) _release/finalize

## release/major - bump the major version, date the CHANGELOG section, commit and tag v<version> (local only)
release/major: _release/preflight
	bump2version --no-commit --no-tag major
	$(MAKE) _release/finalize

# A sub-make, so VERSION is the bumped one. The Unreleased entries become the
# version's section under a fresh, empty Unreleased heading; the lock follows the
# workspace version; the tag is made last, on the commit that holds all of it.
_release/finalize:
	$(CARGO) update --workspace
	sed -i 's/^## \[Unreleased\]$$/## [Unreleased]\n\n## [$(VERSION)] - '"$$(date -u +%F)"'/' CHANGELOG.md
	git commit -m "chore: release v$(VERSION)" -- .bumpversion.cfg Cargo.toml Cargo.lock CHANGELOG.md
	git tag -a "v$(VERSION)" -m "Release v$(VERSION)"

## release/notes - write the CHANGELOG section of v<version> to target/release-notes-v<version>.md (the GitHub release body)
release/notes:
	mkdir -p target
	awk -v head="## [$(VERSION)]" 'index($$0, "## [") == 1 { if (on) exit; on = index($$0, head) == 1; next } on' CHANGELOG.md >$(RELEASE_NOTES)
	@[ -s $(RELEASE_NOTES) ] || { echo "error: CHANGELOG.md has no section for $(VERSION)"; exit 1; }

## release/publish - push HEAD and the tag v<version> to RELEASE_REMOTE, then package/deb/publish
release/publish:
	git push $(RELEASE_REMOTE) HEAD "v$(VERSION)"
	$(MAKE) package/deb/publish

# ---------------------------------------------------------------------------
##@ Dependencies and tooling
## Local sven/brain checkouts, git hooks, formatting.

## deps - lists the dependency targets; does nothing
deps:
	@bash scripts/make-help.sh deps

## deps/local - build against local sven/brain checkouts (SVEN_DIR, BRAIN_DIR); writes the gitignored .cargo/config.toml
deps/local:
	bash scripts/local-config.sh $(SVEN_DIR) $(BRAIN_DIR)

## deps/lua - build the pinned Lua 5.1.5 into LUA_DIR/bin/lua from its checksummed source tarball; run tasks with --runtime-dir LUA_DIR/bin (LUA_DIR=<directory>)
deps/lua:
	test -n "$(LUA_DIR)" || { echo "deps/lua: give LUA_DIR=<directory to build into>"; exit 2; }
	bash scripts/deps/lua51.sh "$(LUA_DIR)"

## deps/tlaplus - install the pinned TLA+ tools (tla2tools.jar) and an aarch64 Temurin JRE into TLAPLUS_DIR, both sha256-checked; TLC runs as TLAPLUS_DIR/bin/tlc (TLAPLUS_DIR=<directory>)
deps/tlaplus:
	test -n "$(TLAPLUS_DIR)" || { echo "deps/tlaplus: give TLAPLUS_DIR=<directory to install into>"; exit 2; }
	bash scripts/deps/tlaplus.sh "$(TLAPLUS_DIR)"

## deps/lock - pin Cargo.lock to the HEADs of SVEN_DIR and BRAIN_DIR (push those commits before sharing the lock)
deps/lock:
	bash scripts/pin-lock.sh $(SVEN_DIR) $(BRAIN_DIR)

## qualify - lists the qualification targets; runs nothing
qualify:
	@bash scripts/make-help.sh qualify

## qualify/list - validate the qualification plan and list every check and mandatory ablation (runs nothing)
qualify/list:
	python3 scripts/qualify/qualify.py scripts/qualify/plan.toml --out "$${TMPDIR:-/tmp}/qualify-list" --list

## qualify/test - the qualification harness's own spec (no model, no GPU, seconds)
qualify/test:
	python3 scripts/qualify/test_qualify.py

## qualify/run - run the qualification matrix and the ablations once, recorded under OUT (OUT=<dir>, VARS="--var splinter=... --var card=..."; the evidence stays out of git)
qualify/run:
	test -n "$(OUT)" || { echo "qualify/run: give OUT=<directory for the evidence>"; exit 2; }
	python3 scripts/qualify/qualify.py scripts/qualify/plan.toml --out "$(OUT)" $(VARS)

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
.PHONY: help build build/crate $(BUILD_CRATE_TARGETS) test test/crate $(TEST_CRATE_TARGETS) test/e2e test/e2e/brain-sven test/e2e/turn test/e2e/turn/spec test/families test/package fmt \
	package package/deb package/deb/release package/deb/publish release release/patch release/minor release/major release/notes release/publish _release/preflight _release/finalize \
	check check/fmt check/clippy check/papers check/gates check/gates/text-hygiene check/gates/spdx $(GATE_TARGETS) \
	lease lease/test lease/doctor lease/gpu lease/gpu/run lease/card lease/card/run lease/worktree lease/worktree/add lease/worktree/list lease/worktree/remove \
	deps deps/local deps/lua deps/tlaplus deps/lock \
	hooks hooks/install \
	qualify qualify/list qualify/test qualify/run \
	papers papers/pdf $(PAPER_DIRS) $(PAPER_TARGETS) \
	samples samples/generations samples/generations/anchor-200 samples/tool-syntax samples/tool-syntax/audit samples/repo-replay samples/repo-replay/audit samples/model-check samples/model-check/audit samples/model-check/report \
	docs docs/build docs/check docs/pdf docs/clean _docs-publisher-pin \
	agents agents/roadmap agents/roadmap/index
