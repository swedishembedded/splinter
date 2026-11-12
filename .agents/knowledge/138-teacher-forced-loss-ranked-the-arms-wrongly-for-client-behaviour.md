<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 138. Teacher-forced held-out loss ranked the arms wrongly for client behaviour

On the five client-behaviour corpora the held-out teacher-forced loss of the tuned model was 0.354 for training on everything and 0.635 for selected material (lower is better; random 2.246; base 5.5 to 6.0), which ranks everything first. Generated exact match on the same questions ranked them the other way: selective 0.400 against everything 0.133 (paired +0.267 [0.067,0.500]). On the property form the two metrics agree (loss 0.279 against 0.797, exact match 0.158 against 0.008). The behaviour values come from a closed set of 16 phrases: a model trained on all the statements learns that distribution and puts probability mass on the right phrases without choosing the right one for the question, which teacher-forced loss rewards and generation does not.

Rule: judge a training arm by generated answers checked by a verifier, not by teacher-forced loss, wherever answers come from a small closed set. Evidence: `~/project/shared/goal-selection/weights2/result-82*-*.json` against `exam-analysis.txt`.
