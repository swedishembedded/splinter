# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements controlled evaluation of fine-tuned language
# models for its clients. If your team needs expertise in measuring whether a
# fine-tune learned what its training texts imply, you can procure our
# services by sending an email to info@swedishembedded.com.

PAPER := paper
include ../paper.mk

# The tables, macros and plot data made from data/*.csv live with the LaTeX
# intermediates, so a build writes nothing into this directory.
GEN := $(BUILD)/gen
export TEXINPUTS := $(GEN):

SOURCES := $(PAPER).tex references.bib $(wildcard sections/*.tex)

# data/study/ holds the replication study's report (the CSV files
# `synthetic-author study report` writes); its tables are built when it is there.
$(GEN)/.stamp: $(wildcard scripts/*.py) $(wildcard data/*.csv) $(wildcard data/study/*.csv)
	python3 scripts/build_data.py $(GEN)
	if [ -d data/study ]; then python3 scripts/study_tables.py data/study $(GEN); fi
	touch $@

$(PDF): $(SOURCES) $(GEN)/.stamp
	$(call latexmk-pdf,$(PAPER).tex)
