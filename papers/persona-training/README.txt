SPDX-License-Identifier: CC-BY-4.0
Copyright (c) 2026 Martin Schr\"oder <info@swedishembedded.com>
(The license covers this directory: the paper, its data tables and scripts.)

Persona training paper: learning an author's judgment from their writing, a
controlled study with synthetic authors.

Build (needs python3, pdflatex, bibtex and latexmk; no Python packages):

    make          builds target/papers/persona-training.pdf
    make clean    removes the PDF and the LaTeX intermediates

The build writes nothing into this directory. Its generated tables, macros and
plot data go to target/papers/.build/persona-training/gen.

Every number, table and plot in the paper is made from data/*.csv by
scripts/build_data.py (standard library only). It computes the Wilson
intervals, the exact one-sided sign tests and the Newcombe intervals, checks
that the tables are consistent with one another (cells add up to arms,
discordant counts agree with the right counts, the request and rejection
counts of each corpus reconcile, records match decisions, the audit covers the
decisions kept, trained and held-out records add up, the echo counts cover the
kept essays, the error counts cover every item and agree with the tuned arm)
and that the derived
values equal those recorded when the examinations were run, and stops with a
message otherwise.

Data:

    arms.csv, cells.csv, paired.csv   made from the examination outputs with
        python3 scripts/collect.py EXAMS_DIR data 101,202,303
        where EXAMS_DIR holds author-<seed>/exam/<arm>.jsonl
    cross.csv                         the output of `synthetic-author cross`
    audit.csv                         the output of
        python3 scripts/audit.py PLAN_JSONL DECISIONS_JSONL
        with PLAN_JSONL from `synthetic-author plan --seed S --articles 60`
    corpora.csv                       counts from each corpus directory
                                      (articles/, rejected.jsonl, usage.json,
                                      decisions.jsonl, dataset/train.jsonl)
    training.csv                      the training split, the step-240 log
                                      line and, for seed 101, training.json
    authors.csv                       the output of `synthetic-author persona`
    echo.csv                          the output of
        python3 scripts/echo.py SEED SHOW_JSON ARTICLES_DIR [...]
        with SHOW_JSON from `synthetic-author show --seed S`
    confusion.csv                     the output of
        python3 scripts/confusion.py ITEMS_DIR EXAMS_DIR tuned 101,202,303
        with ITEMS_DIR/items-S.jsonl from `synthetic-author items --seed S`
    cost.csv                          the token estimates of usage.json
    spend.csv                         the provider's reported daily spend of
                                      the account before and after each
                                      corpus run

The commands that produce the examination outputs are listed in the paper's
Reproducibility section (samples/synthetic-author).

The walk-through section and the appendix quote the data of one case of
author 101 verbatim (invented years written [year], long dashes written --,
shortenings marked [...]); those listings are typed into the sections, not
generated, because the corpus is not kept in the repository.

Layout: paper.tex (preamble, abstract), sections/ (one file per section),
references.bib, data/, scripts/.

Replication study data:

    python3 scripts/study_data.py REPORT_DIR OUT_DIR [--training-seed-index N]
        turns the `report/` directory of `synthetic-author study report` into
        arms.csv, cells.csv and paired.csv for the N-th training seed of each
        author. build_data.py still reads the three authors of data/; the
        tables made here are what it will read once the paper is laid out for
        the study's authors. The other tables (authors, audit, corpora,
        training, cost, spend, echo, confusion, cross) keep their producers
        above.
