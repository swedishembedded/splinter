# 012. A synthesizer fine-tune that never saw text trained nothing usable

brain's Qwen3-TTS `finetune_lora` and `finetune_full` are a plain language-model
loop over a token file of "text then codes" made by a data generator. The
Talker does not read tokens: at every step it reads one embedding row (the
first codec codebook of the frame just made, plus the sum of the residual
codebooks' embeddings, plus the next row of the projected text stream or the
pad row), after a prompt of rows built from the speaker's x-vector and the text.
Training on the token file therefore teaches a different model than the one that
speaks, and no result of it can be loaded back to change how a sentence sounds.

Training the Talker as a policy needs the rows a rollout was fed, rebuilt from the
codes alone. They were checked three ways before anything was trained on them:
40 of 40 sampled first-codebook tokens are the argmax under the rebuilt rows;
the trainer's final-norm logits and the inference path's agree to 6e-5; and
after a LoRA step the adapter folded into the inference Talker in place gives
logits that match the trainer's to 6e-5 (the fold changed the inference logits by
up to 10.6, so the check is not vacuous).

The step itself is the group-relative policy gradient written as a per-token weight
on the log-likelihood, with the head on the host: a sampled token's gradient is
`(-A + beta (1 - e^u)) (onehot - softmax)`, `u` the reference minus the policy
log-probability. A learning rate that is fine for a credited take (1e-3 raised its
mean log-probability from -0.49 to -0.01 in a few steps) sends the penalised take's
likelihood to the floor and the weights to NaN at 5e-3, because pushing a sampled
token down has no bound. The pairs of a group are updated together and the
learning rate stays near 1e-4.

Two numbers to carry: the free voice of the 0.6B base already holds one speaker
across sampling seeds (embedding similarity 0.95 to a reference), and a voice
pinned to a reference holds it tighter (0.996), so seeds vary delivery, which is
what preference learning needs.
