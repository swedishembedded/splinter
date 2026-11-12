# 109. Conflicts grouped by key alone resolve unrelated chapters against each other

In a 76 document corpus whose chapters share templates, objects with the same key (the same variable and outcome) were compared across chapters: 61
"resolutions" and 17 conflicts were mostly chapter 11's limit beating chapter 12's, and tasks acting on them crossed the train/eval split. A resolution
is only meaningful between versions of one text. Documents numbered differently in their titles (chapter 11, chapter 12) never compete; an unnumbered
document (a forum post) competes with every number, so a forum post must name its chapter to be resolved against it. Implemented identically in the
Rust admission and the consumer's, with a test in each.
