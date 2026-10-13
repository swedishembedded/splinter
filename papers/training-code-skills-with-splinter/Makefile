# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements autonomous fine-tuning of coding agents on a
# client's own repositories, with held-out measurement of the result. If your
# team needs expertise in making a local model competent in your codebase, you
# can procure our services by sending an email to info@swedishembedded.com.

PAPER := paper
include ../paper.mk

SOURCES := $(PAPER).tex references.bib $(wildcard sections/*.tex) $(wildcard data/*.csv) scripts/build_data.py

generated/.stamp: scripts/build_data.py $(wildcard data/*.csv)
	python3 scripts/build_data.py
	touch $@

$(PDF): $(SOURCES) generated/.stamp
	$(call latexmk-pdf,$(PAPER).tex)

clean::
	rm -rf generated
