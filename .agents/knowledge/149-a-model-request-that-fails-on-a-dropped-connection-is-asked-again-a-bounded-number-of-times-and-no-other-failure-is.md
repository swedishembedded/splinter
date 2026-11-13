<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 149. A model request that fails on a dropped connection is asked again a bounded number of times, and no other failure is

**Lesson:** The typed calls (the `lineage` proposer among them) and the solves classified "openai request failed: ... connection
closed before message completed" as transient and then ended the whole generation with it: sven repeats a transient failure
inside an agent turn only, and a caller that runs with its own deadlines turns that repeat off. The one place every model call of
these paths is made is the provider they are handed, so `splinter_agent::retry::Retrying` wraps it there: connection-level
failures and 502/503/504 are asked again up to 4 times, from a pause of 0.5 s that doubles; a refusal, a malformed request or a
deadline is returned at once, and the last error is returned when the retries run out.

## What this does and does not establish

Verified with a scripted provider that drops its first request (it succeeds on the second), one that always drops (3 asks for 2
retries, the error returned), and one that answers HTTP 400 (1 ask). Not measured against a live `brain serve` restart. A build
made before this change does not retry; a running experiment keeps its pinned build.
