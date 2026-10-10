# Roadmap: what the voice agent paper still has to prove

The paper `papers/learning-voice-agent` contains only what is proven by a stored
run (`data/runs/*.json`, listed in `data/runs.csv`). Everything below is a claim
the paper does not make yet. Each item says what finishing it needs, how it is
measured, and whether it can run unattended. When an item passes its gate, add its
run to `data/runs.csv`, its table to `scripts/build_data.py`, and its text to the
paper; when it fails, add the failure to the paper as a result.

Rules: a pre-registration file goes in `papers/learning-voice-agent/prereg/` and is
committed before the run; the held-out split of a set is run once per recipe;
GPU jobs run under `flock ~/project/shared/containers/gpu.lock`; voices are always
labelled synthetic portrayals; nothing unmeasured is stated as a number.

## A. Can run unattended

| Id | Claim to prove | Work | Gate |
|---|---|---|---|
| A1 | Procedures and rules are taught reliably | New scenario set (generator in `scripts/teaching_set.py`): at least 8 new procedures (3 asks each) and 8 new rules (3 asks each) with unseen vocabulary; run dev, then test once | lower 95% Wilson bound of the taught pass rate >= 0.80 per kind, untaught < 0.20 |
| A2 | Several rules taught together are all obeyed | Dev findings so far (`resources/speech/teach/dev-rules-{1,2,3}.json`, 4 scenarios x 4 asks): with rules rendered one per line, 8 of 16 asks passed (over+copy 4/4, rules3 4/4, captain+copy 0/4, over+captain 0/4); a numbered list gave the same 8/16; quoting the user's words (`The user told you: "..."`) gave 4/16. The captain rule alone is obeyed; it is dropped when combined, so the cause is not the rendering alone. Next: try a per-rule check turn (the agent lists the rules it must follow before it answers), splitting rules by kind, a larger model; pre-register the change on new dev scenarios; test split once | every rule obeyed in >= 90% of asks over at least 8 two-rule and 4 three-rule scenarios |
| A3 | The same holds on another model | Run the pre-registered test split of `prereg/teaching-test.md` unchanged on Qwen3-14B and Qwen3-4B (`run-teach-eval.sh` with another `--base`) | report the per-kind rates; claim only for the models that pass the gates |
| A4 | Teaching survives recognition | Render all authored utterances (`data/teaching/utterances.jsonl`) and all teach utterances of the scenario sets in at least 4 synthetic voices (seeds) with `speak-set`, recognise them, run the readers and the scenarios on the recognised text | read rate and scenario pass rate by voice with intervals; list the recognition errors that broke a teaching |
| A5 | Teaching by voice is a rate, not a demonstration | Scale the spoken conversations of `prereg/teaching-spoken.md` to at least 30 scripted conversations across at least 4 voices, same pass conditions | pass rate with a Wilson interval; failures listed by cause |
| A6 | Selection of what the model is told scales | Ledgers of 50, 200 and 1000 facts; ask one fact; compare overlap selection with an embedding selection (`Qwen3-Embedding-0.6B` is in the store) | recall of the right fact in the top k and answer pass rate by ledger size |
| A7 | Procedures are carried out, not only told | Fixture workspace (counts, reads, edits) with a procedure taught by voice; the agent is asked to do the task with sven's tools; the result is checked by script. First measure the persona-free agent's tool reliability on the same tasks without teaching | task success taught vs untaught vs the procedure typed into the system turn |
| A8 | Latency of the teaching path | At least 3 repeats of at least 30 spoken turns on a card shown idle before and after, with and without the ledger and the model reader | first sound p50 <= 1.5 s and p95 <= 2.0 s with intervals, or the miss reported |
| A9 | The planted-reward optimiser comparison is not one seed | Repeat GRPO, DPO and best-of-n of `speech-rl` on at least 3 seeds and on 24 held-out sentences | paired comparison of final frames per word with an interval; WER guard |
| A10 | Pronunciation lessons fix the word and generalise | E1: screen the words the synthesizer gets wrong with two recognisers; teach each once as a lexicon lesson; test in unseen carrier sentences, with untaught words as the no-regression arm | recogniser-agreed correct-pronunciation rate rises; TOST no regression on untaught words. The listening confirmation needs the user (B3) |
| A11 | The reader on text in other phrasings | A second authored utterance set written by a fresh agent (not from the implementer's list) given only the spec of teachings; score the frozen readers on it once | found share and false readings with intervals |
| A12 | Independent review | A fresh Sonnet agent given the paper and the run files, not our conclusions, checks every number against the run files and every claim against its gate | findings recorded and fixed |

## B. Needs the user

| Id | Work | What the user does |
|---|---|---|
| B1 | Reader on real phrasings | Write 50 real corrections and 50 things said to an assistant that are not corrections, in their own words (typed is fine); kept local until they say otherwise |
| B2 | Listening study | Finish the remaining ratings of the artifact session (35 of 100 left); then the judge is refit (`voice-loop judge`) and reported with the real n |
| B3 | Human confirmation of pronunciation (A10) and of any naturalness claim | Blind A/B on a sampled subset of the frozen exam |
| B4 | Real speech | Record about 100 short prompts (questions, directives, corrections; quiet and noisy); transcripts verified by them; local, named by digest |
| B5 | Naturalness training | Only after B2 and B3: a judge-rewarded run of `speech-rl` against the three baselines on a dev set, then the exam once. If it does not win, the debug ladder of the plan is run and the diagnosis is reported |

## C. Larger programmes, outside this paper unless they pass

- A trained persona adapter that beats the base and the prompted base on the frozen exam (the three existing adapters in `huggingface/samuel-adams-persona-lora` did not pass).
- Speech understanding without a transcript, redesigned (the projector in `huggingface/qwen3-asr-speech-projector` failed its gate).
- Imitation from one spoken example and supervised lessons applied at synthesis time.
- A live microphone and speaker client with barge-in.
