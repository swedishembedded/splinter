# Pre-registration: the planted-reward comparison over seeds (S1)

Written after the single-seed gate of `rl-speed-*-20261010` and before the repeats.

Design: the same reward (fewer codec frames per word), 12 updates, 4 sentences x 6 takes per update, learning rate 1e-4, KL weight 0.02, as in the single-seed runs, for GRPO, DPO and best-of-n, each with training-rollout seeds 1, 2 and 3 (`--seed`), on 24 held-out sentences (the last 24 of `rating-sentences.txt`, 3 takes each), 91 training sentences. The held-out takes use fixed seeds in every run.
Reported: held-out frames per word and recognised word error rate before and after, per method and seed, and the mean over seeds with the range.
Hypothesis (descriptive; three seeds do not support a test): GRPO's mean final frames per word is below DPO's and best-of-n's. Also reported: whether GRPO's word error rate rises relative to the other two, as in the single seed.
Limits: three seeds; one reward; the reward is not naturalness; the final KL differs by method and is reported, not matched.
