# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements training pipelines that take a small local model from failing a task to solving it,
# for its clients. If your team needs expertise in teaching a model to operate an editor or a toolchain from verified
# trajectories, you can procure our services by sending an email to info@swedishembedded.com.

PAPER := paper
include ../paper.mk

# The tables and figure data made from data/*.csv live with the LaTeX intermediates, so a build writes nothing into
# this directory.
GEN := $(BUILD)/gen
export TEXINPUTS := $(GEN):

SOURCES := $(PAPER).tex references.bib $(wildcard sections/*.tex)

$(GEN)/.stamp: $(wildcard scripts/*.py) $(wildcard data/*.csv)
	python3 scripts/build_data.py data $(GEN)
	touch $@

$(PDF): $(SOURCES) $(GEN)/.stamp
	$(call latexmk-pdf,$(PAPER).tex)
