# 022. A stated verifier with a length bound failed correct long answers

Recall tasks were graded by a verifier that accepts an answer stating the
reference, with an upper bound on length. A verifier of formal strength
outranks a judge, so a correct long answer ("There are seven continents: ..."
to a reference of "seven") was failed on length alone, as were a subscript
digit against a plain one and a number in words against its digits. The arm that
gave long worked answers lost on the anchor suite, and the loss was reported as
forgetting. Rescored with a corrected verifier the sign of the drop depended on
how a judge ruled on the answers left to it.

How to avoid it: a code verifier abstains when it cannot decide, and leaves the
answer to a judge; it never fails an answer it could not read. Rescore saved
answers under the corrected verifier before reporting a regression, and count
the answers waiting for a judge as their own category, not as wrong.
Normalise what is compared (sub- and superscript digits, numbers in words).
