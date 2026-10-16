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
	python3 scripts/verify_structure.py
	python3 scripts/build_data.py --check $(if $(EVIDENCE),--evidence $(EVIDENCE))

.PHONY: final
# The release check: as verify, and fails while the paper is marked draft or any
# REGISTERED or PLANNED claim remains (results not yet validated).
final: verify
	python3 scripts/build_data.py --check --final $(if $(EVIDENCE),--evidence $(EVIDENCE))
