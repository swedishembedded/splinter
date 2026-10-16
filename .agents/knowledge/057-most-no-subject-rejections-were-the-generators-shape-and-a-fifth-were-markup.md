# 057. Most no_subject rejections were the generator's shape, and a fifth were markup

A corpus-expansion run over three documentation pages rejected 82 of 84
proposals as `no_subject`. Reading the 82 rejection details against the rule
(the subject must be in the instruction, case and whitespace aside, and named by
the source or a cited section) gives: 18 where the subject's words are in the
instruction but a code quote splits them (subject `q key`, instruction "the `q`
key"); 5 with all the subject's words present but not adjacent; 35 where the
subject the generator named is not in its own question (subject `braintop`,
question "What does the `--cli` option print?"), which the rule rightly refuses;
and 24 where the subject is not literally in the source or section (an invented
label such as `metric line`, or a call written `Run()` where the source says
`Run`). The rule is mostly right; the model's output shape is the main cause.

Only the markup part was a defect: the comparison now ignores code quotes,
quotation marks, emphasis marks and the empty parentheses of a call, in both
the admission of a task and the check of its variants. The subject stays a
literal span of the instruction otherwise, because a question that does not say
what it is about has a different answer for every product.

How to avoid it: before changing an admission rule, classify the rejected items
by reading them against the rule; a count by reason alone cannot tell a strict
rule from a generator that does not follow the brief.
