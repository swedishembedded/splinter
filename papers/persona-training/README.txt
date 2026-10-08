Persona training paper: how Splinter learns to think like a historical author

Build (needs python3, pdflatex, bibtex and latexmk; no Python packages):

    make          builds paper.pdf
    make clean    removes the PDF, LaTeX intermediates and generated/

Every table and plot in the paper is generated from data/*.csv by
scripts/build_data.py (standard library only), which also recomputes the exact
sign tests and the task-level bounds, refuses an exam row whose logged p-value
differs from the exact one and an anchor row whose cells do not add up. To
update a result, edit its row in data/ and run make. Completed runs on the
frozen exam go in data/pilot.csv with state "completed"; the paper prints one
row per run in its pilot tables. Make the row from a run's report with

    python3 scripts/pilot_row.py REPORT --arm NAME --description TEXT \
        --power P --power-fpr F >> data/pilot.csv

which recomputes the counts from the report's per-task records and refuses a
report whose summary disagrees (P and F are the "planned" power and
false-positive rate of `splinter exam-set power --from-report REPORT
--families 39 --tasks-per-family 6 --json`). Runs that examined only the deployed arm
go in data/deployed.csv, paired by task with the prompted arm of a four-arm report:

    python3 scripts/deployed_row.py REPORT --arm NAME --description TEXT \
        --reference REPORT:prompted >> data/deployed.csv

Runs of a candidate on every examinable task of the frozen exam (not a pilot) go in
data/full.csv (the run, its judge, arms and the tasks outside the pilot) and
data/full_comparisons.csv (the primary and the five secondary comparisons):

    python3 scripts/full_row.py REPORT --arm NAME --description TEXT \
        --pilot-report PILOT_REPORT

which appends to both files, recomputes every count and exact test from the
report's per-task records and refuses a report whose summary disagrees;
PILOT_REPORT is the pilot run of the same candidate (its tasks are reported
apart, and its answers are compared with the full run's). build_data.py
recomputes the sign tests, the Holm adjustment and the Bonferroni products and
refuses a row that does not match.

The paper's prose names the runs it discusses and is edited by hand.

Layout: paper.tex (preamble, abstract), sections/ (one file per section),
references.bib, data/ (measured counts with their sources), scripts/.
