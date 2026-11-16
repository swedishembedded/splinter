<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 159. The pretraining text spelled tool-call arguments compactly while the served template spaced them

**Lesson:** text a from-scratch model is pretrained on in its own chat format must be written by the same
serializer the serving and training side uses for every byte, including JSON inside the template. Brain
registers `tojson` as Python's `json.dumps` (`", "` and `": "`), as `transformers` does; minijinja's built-in
`tojson` and `serde_json::to_string` are compact. A spec that renders the template with the built-in filter
agrees with a compact generator and proves nothing about what the model is served.

Found 2026-11-16 in minimind (`samples/minimind/src/template.rs`): the worked trajectories in the
pretraining corpus wrote `{"path":"inputs/a.txt"}`, brain rendered the model's own earlier calls and the
fine-tune records as `{"path": "inputs/a.txt"}`; the pretrained model wrote the compact spelling, the
fine-tuned one the spaced one. `Template::render` now writes arguments with `to_json` (the `json.dumps`
defaults), and the template spec registers that filter and pins a two-key argument rendering.
