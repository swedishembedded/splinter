# 017. A family an earlier run trained on could become a later exam

`practice` keeps the families it reserved for the exam under a pointer and
reserves new ones among the families of the run by a stable hash of their names,
so a family reserved once stays the exam. Nothing remembered the other direction:
the families a run trained on. A second run asked for more exam families than it
had untouched ones, and the hash picked among every family it had not already
reserved - including files the first run had trained a released model on. The
documentation said nothing trained on can be examined later, and the gate's
improvement and retention checks were about to measure exactly that.

The families a run's dataset trains on are now remembered as well, and the
reserve never picks one of them: a run may reserve fewer families than it was
asked for, and says which. The gate also leaves out of an exam any task whose
family the candidate, or a release it descends from, trained on, counted under
`leaked`, so an exam recorded before this change cannot be examined on what its
model learned either. The spec trains what a first run can on a project of six files, reserving
one, and asks the next run, over the same files and one more, for six exam families
of seven: none of the families the first run trained on is in the exam, and the
one it reserved still is. With the remembered set taken out of the reserve, the
spec fails on the first assertion: `shop/tier4.py`, a family the first run trained
on, is in the second run's exam.

How to avoid it: a rule stated in one direction ("what is reserved is never
trained on") needs its converse enforced where the other side is decided ("what
is trained on is never reserved"), and a stored set that grows needs both sets.
