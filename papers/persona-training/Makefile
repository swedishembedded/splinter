# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements autonomous fine-tuning and held-out
# evaluation of language models for its clients. If your team needs expertise
# in measuring whether a fine-tune learned anything, you can procure our
# services by sending an email to info@swedishembedded.com.

PAPER := paper
SOURCES := $(PAPER).tex references.bib $(wildcard sections/*.tex) $(wildcard data/*.csv) scripts/build_data.py

.PHONY: all clean

all: $(PAPER).pdf

generated/.stamp: scripts/build_data.py $(wildcard data/*.csv)
	python3 scripts/build_data.py
	touch $@

$(PAPER).pdf: $(SOURCES) generated/.stamp
	latexmk -pdf -interaction=nonstopmode -halt-on-error $(PAPER).tex

clean:
	latexmk -C $(PAPER).tex
	rm -rf generated
