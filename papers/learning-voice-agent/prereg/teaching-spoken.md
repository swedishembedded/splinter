# Pre-registration: teaching by voice (T4)

Three scripted conversations, the user's turns rendered in one synthetic voice (seed 1) by `speak-set` and recognised by the
streaming recogniser before the agent hears them; the agent is `voice-loop agent --model-reader` on Qwen3-8B under the Samuel Adams portrayal prompt
(a persona prompt, so replies are in character). The files are `resources/speech/teach/spoken/c{1,2,3}.txt`.

Pass conditions, fixed from the first run on and unchanged by the later ones:

- c1 (a corrected fact): the agent restates and asks, keeps on "Yes.", and the answer to "What is the build server called?" contains "atlas".
- c2 (a rule, then taking it back): the answer to the first question after the rule ends with "over"; after "Forget that." the agent says it forgot, and the next answer does not end with "over".
- c3 (a procedure): the answer to "How do I deploy the site?" contains "harbor" and "seal".

History: the first run found that "Forget that." was taken by the pronunciation desk, which had nothing to take back, and never reached the ledger (fixed: it is routed to whichever kind was
kept last). The second run kept the rule off the ledger but the model kept ending with "over" because its own earlier replies in the conversation did. A note to the model that the rule was cancelled did not
stop it (naming the rule to the model made it follow it), and the first attempt to start the conversation over was a no-op because a method was missing from the voice agent's implementation of the trait;
all three are development findings, shown by two dev scenarios that failed and then passed (`scenarios-session-retract-dev.jsonl`). The final fix starts the conversation over when a teaching is taken back.
The numbers of this study are those of the first run with that fix; earlier runs are development runs and are not results.

One run per conversation. Three conversations are a demonstration that the path works end to end, not a rate.
Limits fixed in advance: synthetic speech of one voice, not a person's; three conversations; the persona prompt shapes the replies.
