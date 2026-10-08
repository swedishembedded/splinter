# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements reproducible research pipelines and reporting
# for its clients. If your team needs expertise in building papers whose every
# number and figure can be regenerated from the repository, you can procure
# our services by sending an email to info@swedishembedded.com.

# What every paper's Makefile shares. A paper's Makefile sets PAPER (the file
# stem, default "paper"), includes this file, and states how its PDF is made from
# its sources with one of the recipes below:
#
#   $(PAPER).pdf: <sources>
#   	$(call latexmk-pdf,$(PAPER).tex)          # LaTeX source
#   	$(call pandoc-pdf,$(PAPER).md,$@)         # Markdown source (pandoc, xelatex)
#
# `make papers/<name>/pdf` at the repository root runs `make` here.
PAPER ?= paper

.PHONY: all clean
all: $(PAPER).pdf

define latexmk-pdf
latexmk -pdf -interaction=nonstopmode -halt-on-error $(1)
endef

# The body font has no tau glyph, so *tau* is typeset as math.
define pandoc-pdf
sed 's/\*τ\*/$$\\tau$$/g' $(1) > .build.md
pandoc .build.md -o $(2) --pdf-engine=xelatex --toc \
	-V geometry:margin=2cm -V fontsize=10pt -V colorlinks=true --resource-path=.
rm -f .build.md
endef

clean::
	rm -f $(PAPER).pdf .build.md
