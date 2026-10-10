# Pre-registration: the teaching test on other models (T2 replication)

Written after the main test split had been run on Qwen3-8B and before it is run on any other model.

Question: do the results of `prereg/teaching-test.md` depend on the 8B model?
Design: the same test split (`data/teaching/scenarios-test.jsonl`, unchanged digest), the same code and the same neutral system turn, run once each on Qwen3-4B and Qwen3-14B (non-thinking, CUDA).
Reported: the per-kind rates and the verdicts of H1 to H3 exactly as defined there. Nothing is changed between models; a gate that a model does not meet is reported as not met for that model.
A claim of T2 is made for a model only if it meets the H1 gates for fact and corrected fact and the H2 gates.
Limit: two further models of one family; the procedure gate cannot be met for any model with 6 asks.
