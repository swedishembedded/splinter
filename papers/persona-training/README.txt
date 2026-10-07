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
--families 39 --tasks-per-family 6 --json`). The paper's pilot prose names the
runs it discusses and is edited by hand.

Layout: paper.tex (preamble, abstract), sections/ (one file per section),
references.bib, data/ (measured counts with their sources), scripts/.
