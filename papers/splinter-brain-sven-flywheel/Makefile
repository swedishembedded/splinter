# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements self-improving local model pipelines whose
# every claim is backed by a rerunnable measurement, for its clients. If your
# team needs expertise in turning an agent's verified experience into a
# better local model, you can procure our services by sending an email to
# info@swedishembedded.com.

PAPER := paper
include ../paper.mk

# Everything the build derives from data/ lands next to the LaTeX intermediates,
# not in this directory; LaTeX finds it through TEXINPUTS.
GENERATED := $(BUILD)/generated

SOURCES := $(PAPER).tex status.txt references.bib $(wildcard sections/*.tex) $(wildcard data/*.csv) data/measurements.json \
	scripts/build_data.py

$(GENERATED)/.stamp: status.txt scripts/build_data.py $(wildcard data/*.csv) data/measurements.json
	python3 scripts/build_data.py --out $(GENERATED)
	touch $@

$(PDF): $(SOURCES) $(GENERATED)/.stamp
	mkdir -p $(BUILD)
	TEXINPUTS="$(BUILD):" latexmk -pdf -interaction=nonstopmode -halt-on-error \
		-outdir=$(BUILD) -jobname=$(NAME) $(PAPER).tex
	cp $(BUILD)/$(NAME).pdf $@

.PHONY: verify
# Re-checks every STRUCTURE claim of the ledger against the repositories and
# every MEASURED claim against data/measurements.json, then the ledger against
# the text. The raw logs behind it are kept outside the repository; when their
# directory exists (EVIDENCE, default in build_data.py) their digests and values
# are checked too, when it is absent that check is skipped.
verify:
	python3 scripts/test_verify_structure.py
	python3 scripts/test_build_data.py
	python3 reproduce/test_examiner.py
	python3 scripts/verify_structure.py
	python3 scripts/build_data.py --check $(if $(EVIDENCE),--evidence $(EVIDENCE))

.PHONY: final
# The release check: as verify, and fails while the paper is marked draft or any
# REGISTERED or PLANNED claim remains (results not yet validated).
final: verify
	python3 scripts/build_data.py --check --final $(if $(EVIDENCE),--evidence $(EVIDENCE))

# ---------------------------------------------------------------- reproduction
# Three tiers, each from the public remotes of brain, sven and splinter and the evidence dataset
# (reproduce/dataset.json pins its revision and every file's digest). Downloads and builds land
# under target/papers/, never in this directory. Variables:
#   EVIDENCE_DIR  the dataset copy (default: where `make verify` looks for it); a local copy is used
#                 as is, only files missing or not at their digest are downloaded
#   REPRO         work directory of tiers 2 and 3 (models, builds, grades, runs)
#   BRAIN_SRC, SVEN_SRC  clones of brain and sven (default: sibling checkouts of this repository)
#   BRAIN_BIN     a brain binary to use instead of building the pinned commit
#   DEVICE        the card brain serves on (default gpu0); LOCK a command prefix that holds it
#                 (for example `flock /path/to/gpu0.lock`)
EVIDENCE_DIR ?= $(OUT)/.evidence/$(NAME)
REPRO ?= $(OUT)/.reproduce/$(NAME)
REPO_ROOT := $(abspath ../..)
BRAIN_SRC ?= $(abspath $(REPO_ROOT)/../brain)
SVEN_SRC ?= $(abspath $(REPO_ROOT)/../sven)
DEVICE ?= gpu0
LOCK ?=
BRAIN_BIN ?=
EXAMINE_BRAIN = $(if $(BRAIN_BIN),$(BRAIN_BIN),$(REPRO)/bin/brain-796871647)

.PHONY: reproduce reproduce/verify reproduce/examine reproduce/rerun
## reproduce - lists the reproduction tiers; runs nothing (tiers 2 and 3 need a GPU)
reproduce:
	@echo "reproduce/verify   tier 1, CPU: download the evidence dataset and check every record (minutes)"
	@echo "reproduce/examine  tier 2, one GPU: re-grade the base and the twelve S4 adapters (time not yet measured)"
	@echo "reproduce/rerun    tier 3, one GPU: re-run the four S4 lineages from the base, then tier 2 on them (about 12 h on a GH200)"

## reproduce/verify - tier 1 (CPU, a 1.6 GB download, minutes): fetch the evidence dataset at its pinned revision, check every file, archive member and adapter digest, then `make verify` on it
reproduce/verify:
	python3 reproduce/evidence.py fetch --dir $(EVIDENCE_DIR)
	$(MAKE) verify EVIDENCE=$(EVIDENCE_DIR)

$(EXAMINE_BRAIN):
	bash reproduce/build.sh brain $(BRAIN_SRC) $(REPRO)

## reproduce/examine - tier 2 (one GPU, the paper's runs used a GH200; time not yet measured): the reference examiner re-grades the base and the twelve S4 adapters on brain 796871647 and compares with the paper's verdicts
reproduce/examine: $(EXAMINE_BRAIN)
	python3 reproduce/evidence.py fetch --dir $(EVIDENCE_DIR) --only exam.jsonl records/rsi4.tar.xz adapters/rsi4
	python3 reproduce/sources.py qwen --dir $(REPRO)/models
	$(LOCK) python3 reproduce/examiner.py grade --exam $(EVIDENCE_DIR)/exam.jsonl --evidence $(EVIDENCE_DIR) \
		--series rsi4 --brain $(EXAMINE_BRAIN) --checkpoint $(REPRO)/models/Qwen/Qwen3-4B --device $(DEVICE) \
		--out $(REPRO)/examine
	python3 reproduce/examiner.py analyse --exam $(EVIDENCE_DIR)/exam.jsonl --evidence $(EVIDENCE_DIR) \
		--series rsi4 --plan $(EVIDENCE_DIR)/rsi4/plan.toml --out $(REPRO)/examine
	python3 reproduce/examiner.py compare --aggregate $(REPRO)/examine/aggregate.json \
		--measurements data/measurements.json --out $(REPRO)/examine/compare.md

## reproduce/rerun - tier 3 (one GH200; the four lineages run one after another, about 12 h, as S4's took 2.6 to 3.4 h each on a card of their own): build the S4 lineage driver at splinter 86f13672a with sven fc99a2f1 and brain 796871647, fetch Qwen3-4B and the Lua 5.1 manual, build Lua 5.1.5, run the four lineages with the launch record's settings and seeds, then grade the new adapters as tier 2 does
reproduce/rerun: $(EXAMINE_BRAIN)
	python3 reproduce/evidence.py fetch --dir $(EVIDENCE_DIR) --only exam.jsonl records/rsi4.tar.xz
	python3 reproduce/sources.py qwen --dir $(REPRO)/models
	python3 reproduce/sources.py manual --dir $(REPRO)/manual
	$(MAKE) -C $(REPO_ROOT) deps/lua LUA_DIR=$(REPRO)/lua
	bash reproduce/build.sh lineage $(REPO_ROOT) $(SVEN_SRC) $(BRAIN_SRC) $(REPRO)
	$(LOCK) bash reproduce/rerun.sh $(REPRO)/bin/lineage-86f13672a $(EXAMINE_BRAIN) $(REPRO)/models \
		$(REPRO)/manual $(REPRO)/lua/bin/lua $(REPRO)/rerun $(DEVICE)
	$(LOCK) python3 reproduce/examiner.py grade --exam $(EVIDENCE_DIR)/exam.jsonl --evidence $(REPRO)/rerun \
		--series . --brain $(EXAMINE_BRAIN) --checkpoint $(REPRO)/models/Qwen/Qwen3-4B --device $(DEVICE) \
		--out $(REPRO)/rerun-examine
	python3 reproduce/examiner.py analyse --exam $(EVIDENCE_DIR)/exam.jsonl --evidence $(REPRO)/rerun \
		--series . --plan $(EVIDENCE_DIR)/rsi4/plan.toml --out $(REPRO)/rerun-examine
	python3 reproduce/examiner.py compare --aggregate $(REPRO)/rerun-examine/aggregate.json \
		--measurements data/measurements.json --out $(REPRO)/rerun-examine/compare.md
