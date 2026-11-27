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

# The tables and figure data are made from data/ (the examiner's verdict of the Lua run, and data/vim-transfer/, a copy of
# the aggregate files of the private evidence dataset swedishembedded/source-to-transfer-evidence) into the build
# directory; the paper reads them as generated/* through TEXINPUTS, so a build writes nothing into this directory.
GENERATED := $(BUILD)/generated
VIM_FILES := evaluation/aggregates.json decision/decision-eval-exam.json decision/secondary-eval-exam.json \
	decision/decision-eval-unseen.json decision/secondary-eval-unseen.json \
	material/funnel.json
export TEXINPUTS := $(BUILD):

$(GENERATED)/.stamp: scripts/paper_data.py data/rsi5-groups-G.json data/artifacts.csv data/builds.csv $(addprefix data/vim-transfer/,$(VIM_FILES))
	python3 scripts/paper_data.py data $(GENERATED)
	touch $@

# Every figure is a TikZ/pgfplots picture in figures/<name>-body.tex that reads generated/ and shares figures/style.tex.
# The paper inputs the body, so the figure is typeset as vector graphics in the paper's own fonts; figures/<name>.tex
# wraps the same body as a standalone document, compiled to a vector PDF and a PNG preview at twice 96 dpi, which are
# published with the figure data.
FIGURES := method funnel exams generations records
PREVIEWS := $(foreach f,$(FIGURES),$(BUILD)/previews/fig-$(f).png)

$(BUILD)/figures/fig-%.pdf: figures/%.tex figures/%-body.tex figures/style.tex $(GENERATED)/.stamp
	mkdir -p $(BUILD)/figures
	# twice, so that a legend placed by name resolves
	for pass in 1 2; do cd figures && TEXINPUTS=.:$(BUILD): pdflatex -interaction=nonstopmode -halt-on-error \
	    -output-directory=$(BUILD)/figures -jobname=fig-$* $*.tex > /dev/null \
	    || { tail -40 $(BUILD)/figures/fig-$*.log; exit 1; }; cd ..; done
	! grep -E '^(Overfull|Underfull)|Warning' $(BUILD)/figures/fig-$*.log

$(BUILD)/previews/fig-%.png: $(BUILD)/figures/fig-%.pdf
	mkdir -p $(BUILD)/previews
	gs -q -dSAFER -dBATCH -dNOPAUSE -sDEVICE=png16m -r192 -dTextAlphaBits=4 -dGraphicsAlphaBits=4 -o $@ $<

.PRECIOUS: $(BUILD)/figures/fig-%.pdf
.PHONY: previews
previews: $(PREVIEWS)

$(PDF): $(SOURCES) $(wildcard figures/*.tex) $(GENERATED)/.stamp $(PREVIEWS)
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
