<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 195. A small model reads a worked example as a description of its own task

**Lesson:** text put in a 4B agent's context competes with the task's own files. A knowledge page whose worked example
looks like the task is read as a description of the task, and a generic rule on the page is applied where the task
states its own. Every page must open with a rule that the task's files and stated rules win, every worked example must be
marked as an illustration with invented data, and pages must not restate what tasks state (two sources for one rule cost
the budget). Measure per family: a gain in one domain can hide harm in another.

Seen 2026-11-20 (item 21 S1b, A2s1, the same 48 tasks x 8 with and without the domain pages preloaded):
- **Per domain:** support +0.17, compiled +0.13, legal +0.05, accounting -0.11.
- **Accounting families:** vat-single-rate 0.88 -> 0.50, vat-boundary 0.86 -> 0.43, vat-multi-rate 0.57 -> 0.25 and
  accrual-accrued 1.00 -> 0.12. The task text and the pages stated the same rules; no contradiction existed. The pages
  failed in four ways:
  - the model called the page's example "the task's example" and expected its structure;
  - it "saw" a date in an empty field;
  - it reduced a three-account answer to the example's single account;
  - it applied a cents-versus-currency rule to plain decimals, rescaling everything by 100, and divided a percent rate
    by 100 twice.
- **Cost:** steps per family rose 1.4-4x (accrual-accrued 53 -> 200), and with-pages failures often wrote no answer file.

Knowledge v0.3 (`swedishembedded/knowledge` 94bc175) carries the fixes; item 21 S1c measures them. Related: 191 (a 4B
told to read a collection almost never opens it).
