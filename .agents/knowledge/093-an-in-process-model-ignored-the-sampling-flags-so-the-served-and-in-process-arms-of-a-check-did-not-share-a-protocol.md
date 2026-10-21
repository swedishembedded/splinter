<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 093. An in-process model ignored the sampling flags, so the served and in-process arms of a check did not share a protocol

`--temperature`, `--top-p`, `--top-k` and `--sampling-seed` were written into sven's configuration for a
model reached through a provider (`remote:...`) and nowhere else. A model solving in process
(`policy:...`, `local:...`) always sampled with `AGENT_SAMPLING` (temperature 0.2, top-k 20, no nucleus
cut, a seed drawn per request), whatever the command line said. The serve-agreement check (G8) solves
the same tasks served and in process under one registered protocol (temperature 0.7, top-p 0.8, top-k 20,
a named seed); the in-process side silently ran a different one, so a disagreement could not be told from
a protocol difference.

Fix: `Sampling` carries an optional `top_p` that reaches brain's request, and the context builds the
sampling of a local model from the configuration (`local_sampling`) when the temperature, nucleus cut,
`top_k`, seed or the reasoning switch is asked for; nothing asked leaves `AGENT_SAMPLING`. The
per-response thinking budget is still honoured only by a served model: brain's chat request has no such
field, so an in-process run with `--thinking` is unbounded.
