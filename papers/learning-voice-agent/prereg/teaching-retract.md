# Pre-registration: taking a teaching back inside the conversation that heard it (T2, addendum)

Written after the main test split had been run and before this one was. A development finding (spoken demonstration, and two dev scenarios in `scenarios-session-retract-dev.jsonl`)
was that a model keeps following a rule it has been told to forget when its own earlier replies in the same conversation followed it. The agent therefore starts the conversation over when a teaching is
taken back. The two dev scenarios failed before that change (all six asks) and passed after it (all six asks); they are not a result.

Test set: `scenarios-session-retract-test.jsonl`, 5 scenarios (one rule, four facts), 3 asks for the rule and 2 for each fact (11 asks), asked in the same session after "Forget that."
Gate: every ask passes its check (the taught content is absent). One run; the same model and settings as the main test.
Limit: the rule check is the absence of a word, and a reply may avoid it by chance; the untaught arm is reported beside it and is the control.
