# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements local models that keep learning from verified
# experience, generation after generation, for its clients. If your team needs
# expertise in continual learning judged by an independent examiner, you can
# procure our services by sending an email to info@swedishembedded.com.

PAPER := paper
include ../paper.mk

SOURCES := $(PAPER).tex status.txt references.bib $(wildcard sections/*.tex)

$(PDF): $(SOURCES)
	$(call latexmk-pdf,$(PAPER).tex)
