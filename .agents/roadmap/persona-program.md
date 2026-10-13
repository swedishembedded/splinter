# Persona program: learn to think, act and write like a person

Goal: `splinter "Learn to think and act like Thomas Jefferson from ./jefferson"`
(optionally with a base model and an adapter name) produces an adapter that
decides, acts through tools and writes as that person would on situations and
tools it has never seen, and a paper whose every number comes from the final
method run end to end. The same command, with no per-person code, must work
for any profession with a body of materials.

Update this file in the commit that finishes or changes an item.

## Decisions taken

- Skills proven: write, judge (what the person would decide, with reasons),
  act (tool calls on unseen tools, including refusals). Headline claims are
  on judge and act.
- A win is a task the prompted base cannot solve and the tuned model can.
  Arms: base, base+persona prompt, tuned, tuned+persona prompt, and base with a
  long prompt or retrieval over the same corpus. The persona prompt carries
  style and role only, checked mechanically for leaks against gold answers.
- Mechanical verifiers first: a captured tool-call argument compared as a
  string; a judge model only where no mechanical check exists.
- Exams are frozen and hashed before the final training run, and each is
  reported once.
- Training: augmented rewriting of every source (paraphrase, question,
  reverse question, consequence), principle extraction, positive and opposing
  claims, self-play against corpus-checked answers. Corpus size is set by a
  saturation curve of held-out gain against expanded size, grounded in the
  literature, not by a time budget.
- Opposing persona: user-supplied opposing materials when given, otherwise
  generated diametrically opposed claims. Whether that works is itself a
  measured result (each model decides differently on the same dilemma, in the
  direction of its author).
- 8B policy, LoRA only. Full fine-tuning and larger orchestrators are out of
  the paper until validated on suitable hardware.
- Selection grammar `base+adapter` everywhere (brain, sven, splinter); an
  existing adapter name continues training.
- Orchestration: a sven orchestrator agent drives the campaign through tools;
  sibling agents (separate engines, separate providers) do preprocessing,
  step checks and judging. Sven wraps the model, brain runs it, splinter
  owns the training machinery.

## Workstreams, in order

1. **Trace and speed.**
   - Per-call generation log (prompt, raw reply, tokens, latency, repairs,
     window and section) and judge rationales stored with verdicts.
   - Task generation is serial today: one model call at a time, up to 6144
     output tokens, 8 tasks per call, one call per kind per window, a
     repair round on malformed replies. Batch across windows and kinds,
     share the prefix across kinds, cap output, drop dead kinds early.
   - Profile brain end to end (prefill vs decode, prefix cache, KV pool
     sizing, adapter folding per engine) before any kernel work; tensor-core
     paths on Hopper only where measured.
   - A sonnet agent reads the full trace of a small real run and reports.
2. **Persona model of the author.** Principle extraction with provenance,
   positive and opposing claims, decision records mined from documented
   decisions (Jefferson: bank opinion, Kentucky resolutions, Louisiana,
   embargo, the Anas; Adams: resolves, committee of correspondence, circular
   letter), opposing material from the founding-america set when relevant.
3. **Exams.** Judge and act exams from documented decisions; a solvable
   programmatic puzzle exam over facts, names and reasoning with a solver
   proving solvability before use; a base-fails task set screened against
   the prompted base; unseen-tool acting exam; retention and anchor suite;
   leak check. Sample shown to the user before freezing.
4. **Brain and sven changes.** `base+adapter` parsing and lazy adapter
   registration; CLI adapter training continues an existing adapter; a
   retrieval store seam in sven (requirements below).
5. **Orchestrated learn.** `Intent::Learn` carries model and adapter;
   `LearnArgs` gains base and adapter; same-name alias continues training;
   an orchestrator role; sibling agents; a trace of the whole run.
6. **Chat sample.** One sample trains and chats: sven chat loop with
   caller-defined tools (`decide(choice, motivation)`), batch mode printing
   the trace, forced-call handling when the model answers in prose.
7. **Final runs and paper.** Jefferson, Adams, their opposites and one
   non-historical profession through the identical command; the paper is
   rewritten from the results (see below).

## Paper rewrite rules

One arXiv-style paper in `papers/persona-training/`. It states the method and
the measured results of the final system only: no development history, no
defect catalogue, no superseded exams or version names, no hardware section
beyond a setup line, no pilots, no "does not claim" section (the evidence
must carry the claims). Figures for the pipeline, the exam arms and the
results. Lessons from the old defects move to `.agents/knowledge/`. Drafts by
sonnet, final grounding by an opus agent after the writer skill's checks.

## Sven memory requirements (for the planned memory system)

Embedder seam (embeddings are dormant today); bulk ingest with chunking; a
per-host store injected through the engine; typed metadata filters, score
threshold and rerank; citation spans with offsets; a read-only retrieval tool
with fixed top-k and token budget; corpus snapshots with a version id.

## Open prerequisites

- An OpenRouter key for a remote orchestrator (none configured; a local model
  plays the role until then).
- Whether a GH200 will be available for any run in the paper.
