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

SOURCES := $(PAPER).tex references.bib $(wildcard sections/*.tex) $(wildcard data/*.csv) \
	$(wildcard data/runs/*) scripts/build_data.py

$(GENERATED)/.stamp: scripts/build_data.py $(wildcard data/*.csv) $(wildcard data/runs/*)
	python3 scripts/build_data.py --out $(GENERATED)
	touch $@

$(PDF): $(SOURCES) $(GENERATED)/.stamp
	mkdir -p $(BUILD)
	TEXINPUTS="$(BUILD):" latexmk -pdf -interaction=nonstopmode -halt-on-error \
		-outdir=$(BUILD) -jobname=$(NAME) $(PAPER).tex
	cp $(BUILD)/$(NAME).pdf $@

.PHONY: verify
# Re-checks every STRUCTURE claim of the ledger against the repositories and
# every MEASURED claim against the raw runs kept in data/runs, then the
# ledger against the text.
verify:
	python3 scripts/verify_structure.py
	python3 scripts/build_data.py --check
