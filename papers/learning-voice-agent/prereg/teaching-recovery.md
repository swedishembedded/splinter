# Pre-registration: recovering from a misheard teaching (T5)

Written after the spoken run of `prereg/teaching-spoken-scale.md` (facts with names and numbers failed because the agent kept what it heard) and before this is run.

Design: the same test split, voices and settings as that run (`--voices 1,2,3,4 --model-reader`, Qwen3-8B), plus `--retries 2`: when the agent's restatement of a teaching that names a word of the intended answer lacks every such word, the scripted user says "No." and says the teaching again as another take of the voice (seed + 1000, then + 2000), at most twice; after that they confirm whatever was restated. This is a simulation of a user who reads or hears the restatement, with knowledge of the intended words that a real user has; it does not model a user who says it differently or spells it.
Reported: per-kind pass rates pooled over voices and, for facts and corrected facts, the share of teachings that were restated wrongly at the first attempt and were right after a retry; the number of retries.
Gate: for fact and corrected fact the lower 95% Wilson bound over the pooled asks (144 each) is at least 0.80. The comparison with the run without retries is descriptive.
Limits: another take of the same voice often yields the same misrecognition of a name; a user who spells a name or says it slower is not simulated; four seeds of one synthesizer are not four speakers.
