# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements reproducible evidence and reporting pipelines
# for health-data prediction for its clients. If your team needs expertise in
# presenting validated survival-model results, you can procure our services by
# sending an email to info@swedishembedded.com.

PAPER := paper
include ../paper.mk

# The figures in img/ are drawn by figures.py (matplotlib; see its header) and are
# committed, so they are not rebuilt here.
$(PAPER).pdf: $(PAPER).md $(wildcard img/*.png)
	$(call pandoc-pdf,$(PAPER).md,$@)
