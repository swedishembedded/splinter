<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
-->

# persona-chat

Talk to a model, a trained persona or not, through tools you define, and read
what it said and called.

```text
persona-chat chat  --model Qwen/Qwen3-8B+maren [--tools tools.json | --options a,b] [--trace trace.jsonl]
persona-chat batch --model Qwen/Qwen3-8B+maren --prompts questions.jsonl [--tools tools.json] --trace trace.jsonl
```

`--model` is a name in the model store, `+` an adapter trained on it (the one
`splinter learn --adapter maren` published), or any reference the other
commands take (`local:...`, `remote:...`, with `--allow-remote`). Add `@16384`
to limit the context to what fits the card.

## Tools

Without `--tools` the model is offered `decide(choice, motivation)` over the
options of `--options`. A tools file is a JSON list:

```json
[{"name": "decide", "description": "Record the decision you take.",
  "parameters": [
    {"name": "choice", "type": "enum", "values": ["capitalism", "communism"]},
    {"name": "motivation", "type": "string"}],
  "result": "Recorded.", "final": true}]
```

Types are `string`, `number`, `boolean` and `enum`; an argument is required
unless it says `"required": false`. Every call is checked against its tool. A
missing argument, a value outside the options or a tool that does not exist is
answered with a message saying so, which the model may correct itself by; a
correct call is answered with the tool's `result`. A correct call to a tool
marked `"final": true` ends the exchange; otherwise the model gets another
turn, up to `--rounds`.

## Trace

Each exchange is one line of the trace file (`id`, the `model`, the `decoding`
(temperature, top-k and the seed of the exchange's draws), the prompt, and per round
the words, the calls with their arguments, whether each was valid and what it
was answered, tokens and milliseconds), and is printed as

```text
> A matter arose over the Solbrook granary. ...
  round 1
    called decide({"choice":"act_alone","motivation":"..."}) -> Recorded.
```

A model that runs here samples, so no two runs are alike unless they share a
seed. `--seed` is the session's seed; without it one is drawn and printed on
stderr. Each exchange is decoded from a seed derived from the session seed and
its position, and its trace line records that seed: passing the session seed
back repeats a session, and the exchange seed in a line is the decoding to
replay for that exchange. A model reached over a network (`--allow-remote`)
samples as its server does, and its lines carry no `decoding`.

A batch file is JSON lines `{"id": "...", "prompt": "..."}` or plain lines;
each question is put to the model on its own.
