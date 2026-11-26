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

# The Vim section's tables and figure data are made from data/vim-transfer/, a copy of the aggregate files of the
# private evidence dataset swedishembedded/source-to-transfer-evidence, into the build directory; the paper reads them
# as generated/* through TEXINPUTS, so a build writes nothing into this directory.
GENERATED := $(BUILD)/generated
VIM_FILES := evaluation/aggregates.json decision/decision-eval-exam.json decision/secondary-eval-exam.json \
	material/funnel.json
export TEXINPUTS := $(BUILD):

$(GENERATED)/.stamp: scripts/vim_transfer_data.py $(addprefix data/vim-transfer/,$(VIM_FILES))
	python3 scripts/vim_transfer_data.py data/vim-transfer $(GENERATED)
	touch $@

$(PDF): $(SOURCES) $(GENERATED)/.stamp
	$(call latexmk-pdf,$(PAPER).tex)

.PHONY: vim-data
# Refresh data/vim-transfer/ from a snapshot of the evidence dataset:
#   hf download swedishembedded/source-to-transfer-evidence --repo-type dataset --local-dir DIR
#   make vim-data EVIDENCE=DIR
vim-data:
	test -n "$(EVIDENCE)"
	$(foreach f,$(VIM_FILES),install -D -m 644 $(EVIDENCE)/$(f) data/vim-transfer/$(f) &&) true

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
	python3 scripts/test_rsi6_criteria.py
	python3 $(ARXIV_TOOL_TEST)
