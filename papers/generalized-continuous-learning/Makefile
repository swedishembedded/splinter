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

# The arXiv source archive, built with the PDF and checked by compiling it as arXiv does (papers/tools/make_arxiv.py).
all: $(ARXIV)

$(ARXIV): $(PDF) $(ARXIV_TOOL)
	python3 $(ARXIV_TOOL) build --paper . --build $(BUILD) --job $(NAME) --out $@
	python3 $(ARXIV_TOOL) check --archive $@ --main-log $(BUILD)/$(NAME).log --work $(BUILD)/arxiv-check

.PHONY: arxiv arxiv/check
arxiv: $(ARXIV)

# Rebuilds the archive from the last PDF build and checks it in a clean room.
arxiv/check: $(PDF)
	python3 $(ARXIV_TOOL) build --paper . --build $(BUILD) --job $(NAME) --out $(ARXIV)
	python3 $(ARXIV_TOOL) check --archive $(ARXIV) --main-log $(BUILD)/$(NAME).log --work $(BUILD)/arxiv-check

clean::
	rm -f $(ARXIV)

.PHONY: verify
# The analysis scripts' tests (the criteria of prereg/rsi5.md as read from the examiner's group verdict).
verify:
	python3 scripts/test_rsi5_criteria.py
	python3 $(ARXIV_TOOL_TEST)
