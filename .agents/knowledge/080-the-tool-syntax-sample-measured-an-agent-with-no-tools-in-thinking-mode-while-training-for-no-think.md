<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 080. The tool-syntax sample measured an agent with one tool, in thinking mode, while its data was for no-think mode

`samples/tool-syntax` built its engine with `Engine::builder().config(..).build()`. An embedded
sven engine offers no built-in tools unless the application asks (`Toolset::none()` is the
default), and under the default deferred disclosure the only definition a request then carries
is the `tools` tool. The first request of an episode on the pinned sven (f0cfb82c) held 1 tool
(`tools`); the model's "calls" were `tools list` and `tools load shell`, answered "No tools are
waiting to be loaded" and "Not loaded: shell (not a tool)". A baseline taken that way measures
a model that cannot reach `shell`, whatever it learned, and a before/after number built on it
would have said so in neither direction.

Two more gaps sat beside it. The request carried no `chat_template_kwargs.enable_thinking:
false`, so the server rendered the thinking prompt (the transcripts show a leading `"\n\n"`
assistant message, the emptied reasoning block), while `train` trains for no-think mode: the
adapter would have been scored in a mode it was never trained for. And nothing enforced the
family's declared `limits.tool_calls`, so a looping base model ran until the context overflowed
(one episode: "needs 12338 tokens, capacity 12288", an error that would have been dropped from
the denominator).

Fix: `agent::run_agent` builds the engine with `Toolset::coding()` under sven's default
deferred disclosure (8 definitions: `edit_file find_file grep read_file shell todo tools
write_file`, 3428 prompt tokens), configures the model with `RemoteModel::apply_to` with
thinking off (the configuration Splinter's own served arms use) plus an explicit per-episode
sampling seed, and cancels the run past the family's tool-call budget. Spec:
`samples/tool-syntax/tests/agent_offers_the_working_tools.rs` asserts the first request's tool
names, `enable_thinking: false` and the seed at the wire.

A fourth defect sat in the demonstrations. `Episode::start` kept the run directory as given, so
a relative `--run-dir` made `UPLOAD_SERVICE_SOCKET` relative to the process while the tools run in
the workspace: every `./svctl` call of every demonstration answered "[exit 1] cannot reach the
upload service", and the 8 demonstrations still came out "8/8 verified" because the verifier
looks from outside. The trainer parsed and encoded them without complaint (32 supervised
examples); only reading the tool messages showed it. Now the directory is made absolute, a
socket path over the 107-byte limit is refused by name (it used to surface as "did not create
its socket within 15s"), and `perform` refuses an observation that reports `[exit N]` (N not 0)
or `error:` instead of recording it as a demonstrated step. Specs:
`tests/episode_is_reachable_from_its_workspace.rs` and `perform::tests`.

Not covered: the budget is enforced from the event stream, so under load it can overshoot by a
call (see 076).
