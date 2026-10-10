# Pre-registration: a taught procedure is carried out with tools (T6)

Written before the test split is run; the dev split (2 procedures, 4 asks) is used to see that the harness and the agent's tools work and is not a result.

Design: each scenario teaches one procedure by a spoken-style utterance (create a named file, write one word on the first line, another on the second) and then asks for the job by name in a fresh session that holds the ledger; the agent has sven's coding tools confined to an empty workspace of its own, with approvals automatic. The check is a script: the file exists and holds both words. The untaught arm has the same tools and the same ask and no ledger.
Test set: `data/teaching/scenarios-tools-test.jsonl`: 8 procedures, each asked in two phrasings, one ask per scenario (16 asks). Qwen3-8B, neutral system turn, one run.
Gate: the taught pass rate has a lower 95% Wilson bound of at least 0.80 over the 16 asks and the untaught rate is below 0.20 (16 of 16 gives 0.81, so there is no room for a miss; a miss is reported as the gate not met).
Limits: tiny file tasks with invented words; the model must call a write tool correctly; a procedure that needs reading or editing existing files is not tested; text input; one model.
