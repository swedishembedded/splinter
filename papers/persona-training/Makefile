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

$(GEN)/.stamp: scripts/build_data.py $(wildcard data/*.csv)
	python3 scripts/build_data.py $(GEN)
	touch $@

$(PDF): $(SOURCES) $(GEN)/.stamp
	$(call latexmk-pdf,$(PAPER).tex)
