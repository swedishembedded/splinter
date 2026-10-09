# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements voice agents that learn how to speak from
# their users, with every claim backed by a reproducible measurement, for its
# clients. If your team needs expertise in evaluating spoken dialogue systems,
# you can procure our services by sending an email to info@swedishembedded.com.

PAPER := paper
include ../paper.mk

SOURCES := $(PAPER).tex references.bib $(wildcard sections/*.tex) $(wildcard data/*.csv) $(wildcard data/runs/*.json) scripts/build_data.py

generated/.stamp: scripts/build_data.py $(wildcard data/*.csv) $(wildcard data/runs/*.json)
	python3 scripts/build_data.py
	touch $@

$(PAPER).pdf: $(SOURCES) generated/.stamp
	$(call latexmk-pdf,$(PAPER).tex)

clean::
	latexmk -C $(PAPER).tex
	rm -rf generated
