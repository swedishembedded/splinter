<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Learning methods: one falsifiable smoke each

Scope: a fast, GPU-free smoke per learning method (SFT, rejection sampling, DPO, GRPO, distillation, continued pretraining, test-time search, retrieval, conversation self-play), the toy model and task they share, and the methods that have no smoke yet.

The umbrella is `make test/methods` (`crates/pipelines/tests/methods`, `scripts/methods/table.sh`): it trains and
samples a toy model on the CPU backend and prints one row per method, the held-out score before and after. A smoke
is a gate on the pipeline, not a result about a real model: it proves that the stages are wired so that the method moves a
number it can move, and that a regression would show. The bench numbers on real models are in
`progress/learning-methods.md`.

## The toy world

- The model is brain's `brain::toy` checkpoint (2 layers, 32 wide, byte-level tokenizer, bare chat template that
  stops at `<|im_end|>`), reached through `splinter_model::fixture::toy_chat_checkpoint`. The weights are the standard
  initialisation widened sixfold: at the standard 0.02 the tied output embedding is too small for any adapter to lift a
  logit within a feasible number of steps.
- The task is a Caesar shift of three letters over `a`..`h`. The 512 prompts are shuffled once; a method may learn
  from the first 150 and is scored on the next 100, which it never sees. Exact match, greedy decoding, unless the row says
  otherwise. The verifier knows the cipher; the policy is never shown the rule.
- Methods that improve a partly right model start from the same weak adapter: SFT on 60 prompts for 200 steps
  (held-out 0.39 greedy, 0.03 to 0.14 sampled at T 0.6 to 1). Rollouts are seeded, so a rerun draws the same ones.

## Results (CPU, 2026-11-12)

See the table `make test/methods` prints; the figures of the last full run are recorded in the Log below.

## Findings that changed the code

- GRPO as configured by default did not move the held-out score on this task: two-sided advantages (the
  default) went 0.39 -> 0.38..0.44 at the learning rates that were stable and to 0 at the ones that were not, with
  shaped (partial-credit) rewards, mean-centred advantages, a KL penalty to the start policy and 6 short on-policy
  rounds all tried. The same machinery on the right answers alone (policy gradient with advantage 1 on the
  rejection-sampled records) went 0.39 -> 0.55 in one round, so the objective, the behaviour proxy and the clip are
  sound. A wrong answer shares most of its tokens with a right one, so pushing it down pushes down the right tokens in it.
  Added `--advantage positive` (`AdvantageKind::Positive`): GRPO's standardised advantage for the attempts that beat
  their group, no record for the rest. The GRPO and self-play smokes use it. The two-sided variant is the default
  and is unproven here.
- The policy-gradient objective could not be run from Splinter on rollouts that carry no sampler log-probabilities
  (the records Splinter's views write never do): brain refused every token. `--behaviour start` /
  `PolicyGradientOptions::behaviour` selects brain's start-of-run proxy, exact for on-policy rounds.
- Plain DPO collapsed the toy model (0.39 -> 0.01 at 1e-3 and 5e-4), as likelihood displacement predicts for answers
  that differ in one token. With the anchor on the chosen answer (`nll_weight` 2) and two on-policy rounds it
  gains. The smoke uses the anchor and says so.
- Continued pretraining had no trainer in brain's public SDK, so `dataset build --view cpt` could only export.
  A text is now written as one supervised assistant turn with nothing before it (the next-token loss over every token),
  which `train` reads as any chat dataset. Only the chat template's framing of that turn is not raw pretraining.
- `Answerer::ask_decoding` draws under a seed of the caller's choosing (the answerer's decoding was fixed at load).

## Not proven

- Process (step-level) verification. `sft-step` keeps the good steps of failed attempts from step labels, and
  `AnnotationBody::StepLabel` exists, but nothing writes the labels from a step verifier, and a tool-less trajectory has
  no actions for the view to read (a reply answered by feedback is context, not an action). A smoke needs a task whose
  steps are tool calls, a toy chat template that renders them, and a step verifier stage; none is built.
- Skill library. `skill_planner` and `splinter_knowledge::skills::admit` have no model-level smoke; no learning
  effect of a library is measured.
- Test-time critique and refinement (`critique.rs`): only best-of-n against a check the model can run is smoked.
- Distillation is sequence-level (a teacher's answers to unlabelled prompts). Logit-level distillation and the
  privileged-context self-distillation view are brain features with no Splinter-reachable smoke.
- Agentic RL through tool use, harness/prompt optimisation: out of this lane.
- The smokes use the plain answerer, or a sven agent with no tools (self-play). Rollouts with tools and sandboxes are
  covered by the bench lanes.

## Log

- 2026-11-12: lane created. Nine smokes, all green on CPU in `make test/methods`; figures below.
