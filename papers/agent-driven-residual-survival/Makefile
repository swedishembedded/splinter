# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements reproducible evidence and reporting pipelines
# for health-data prediction for its clients. If your team needs expertise in
# presenting validated survival-model results, you can procure our services by
# sending an email to info@swedishembedded.com.

# Needs pandoc and xelatex. The figures in img/ are drawn by figures.py
# (matplotlib; see its header) and are committed, so they are not rebuilt here.
.PHONY: all clean

all: paper.pdf

# The sans-serif body font has no tau glyph, so *tau* is typeset as math.
paper.pdf: paper.md $(wildcard img/*.png)
	sed 's/\*τ\*/$$\\tau$$/g' paper.md > .paper-build.md
	pandoc .paper-build.md -o $@ --pdf-engine=xelatex --toc \
		-V geometry:margin=2cm -V fontsize=10pt -V colorlinks=true --resource-path=.
	rm -f .paper-build.md

clean:
	rm -f paper.pdf .paper-build.md
