# 019. A retry was shown the output of a check it was never meant to see

The coding-loop sample says a hidden check is "named to the worker and nothing
more". Its feedback did more. After a failed attempt the next one was shown
every failing check as `Check <name> (<command>) failed ...` with the tail of
what it printed, hidden checks included, and the follow-up turns inside an
attempt were built from the same output. The sample's own spec asserted it: the
retry prompt had to contain `AssertionError` or `Traceback` from a check that
was declared hidden. A hidden check's command names the script that holds the
answer, and its output quotes the assertion; both went into the prompt of the
next attempt.

Moving the attempt onto the shared worker and grading made it visible a second
way. The first run of the migrated loop rejected its own second attempt as
`hidden_material_exposed`: the traceback in the feedback carried the path of
the directory the candidate was graded in, which sat under Splinter's state
root, and the exposure audit forbids that path in anything the worker is shown.
The audit was right; the loop was the one that had shown it.

What holds now: only a check the worker is shown (`--accept-visible`) has its
command and output in the feedback and in the follow-up turns, a hidden one is
named and nothing else, and candidates are graded outside the state root.
`a_hidden_check_is_named_to_the_retry_and_nothing_more` counts three strings
(the check script's path, `AssertionError`, a name from the test file) across
every request of a two-attempt run and finds none, and
`feedback_names_a_hidden_check_and_shows_neither_its_command_nor_its_output`
holds the same for the feedback text alone.

How to avoid it: a rule stated in a README is a claim until a spec counts the
forbidden strings in what the model was actually sent. Anything that turns
a verifier's output into a prompt is a channel for the verifier's material.
