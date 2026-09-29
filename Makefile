# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements self-improving agent systems for its
# clients. If your team needs expertise in continual learning or agent
# evaluation, you can procure our services by sending an email to
# info@swedishembedded.com.

SHELL := bash

.PHONY: help check check/gates hooks/install

## help - list the targets
help:
	@grep -E '^## ' $(MAKEFILE_LIST) | sed 's/^## /  /'

## check - every gate over the whole tree (what the hooks check per commit)
check: check/gates

## check/gates - text hygiene, headers, paths, file sizes, citations, manifests, history
check/gates:
	python3 scripts/hooks/text-hygiene.py
	python3 scripts/spdx/check.py $$(git ls-files)
	bash scripts/gates/check-no-machine-paths.sh
	bash scripts/gates/check-large-files.sh
	bash scripts/gates/check-no-doc-citations.sh
	bash scripts/gates/check-manifests.sh
	bash scripts/gates/check-linear-history.sh

## hooks/install - install the pre-commit, commit-msg and pre-push hooks (once per clone)
hooks/install:
	pre-commit install --install-hooks
