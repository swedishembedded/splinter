# 056. A set from one source text was divided, and then refused by its own gate

`learn` on a single fact sheet ended after its rollouts, its teacher solves and
its dataset with "1 group(s) and 0 unit(s) appear in more than one of train,
validation and test", twice, so no release was ever made. Every record of a
source file carries the group of that file's text, so one sheet is one group.
The holdout rule's last resort for "no whole group fits" was to hold out the
newest tenth of the records, dividing the group; the leakage check on the files
about to be written then refused the rule's own output, with counts and no cause.
A spec test even asserted the division.

The rule now holds nothing out when no group fits, and the split of a dataset
file names the cause (`NoGroupToHoldOut`: records, independent groups, add
sources). The held-out measurement needs at least two independent source texts,
more for a paired test (`MIN_HELD_OUT_UNITS`). How to avoid it: give a learning
run one source file per independent unit of knowledge (a directory of sheets),
not one long file; check the number of groups before spending rollouts.
