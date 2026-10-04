# 009. An RL loop needs a measured cost before it is planned

A two-step GRPO cycle (groups of two, 120 new tokens) over a 1.5B model took
1,608 seconds with the GPU shared and ended in a rejection that said nothing,
after the loop had panicked on a Hugging Face directory and trained on rows
the length of the model's whole context (131072 tokens). A cycle long enough
to teach a 7B anything is out of reach on one machine in a session, and a
verifier reward that rewards a valid layout is maximised by the shortest valid
template. Prefer rejection-sampling fine-tuning and on-policy preference
pairs from the same verifier: they use the student's own graded samples and
cost a sampling pass, not a loop.
