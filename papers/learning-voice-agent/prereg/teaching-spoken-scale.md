# Pre-registration: teaching through recognition, at scale (T4)

Written after the three-conversation demonstration of `prereg/teaching-spoken.md` and before this is run.

Design: the unchanged test split of the teaching study (`data/teaching/scenarios-test.jsonl`, 50 scenarios) with every user turn (the teaching, "Yes.", "Forget that.", the asks) said aloud by the synthesizer in four synthetic voices (seeds 1 to 4, one portrayal)
and recognised by the streaming recogniser; the agent gets only the recognised text, and the model reader is on. Qwen3-8B, one run (`voice-loop teach-eval --voices 1,2,3,4 --model-reader`). The untaught arm hears the same recognised asks.
Reported: per-kind pass rates pooled over voices and per voice, the share of teaching utterances not heard as said, and each teaching that was misheard with what was heard and whether the scenario still passed.
Gate: for fact and corrected fact, the lower 95% Wilson bound over the pooled asks (144 each) is at least 0.80 and the untaught rate is below 0.20; the leak and disturbance kinds at least 0.95. A voice that breaks a gate is reported.
Limits: four seeds of one synthesizer are not four speakers; synthetic speech is clean and unaccented; the recogniser is the same family as the one used to verify the voices elsewhere; pooled asks within a scenario and voice are correlated, so intervals over asks are optimistic.
