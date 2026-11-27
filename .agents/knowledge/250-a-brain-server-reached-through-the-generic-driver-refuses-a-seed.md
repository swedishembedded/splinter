<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 250. A brain server reached through the protocol's generic driver refuses every seeded request

**Lesson:** the execution worker configures sven's provider by the connection's wire protocol (`openai`), and sven's
generic OpenAI driver declares no seed control: a rollout that asks for its own seed (`--seed`, which an online run
needs for reproducible draws) is refused before anything is sent. The first online rollouts on the worker all ended
with no model request ("does not honour the sampling control(s) seed"), classified as infrastructure failures, 12 of
12 in the fixture run. A brain server is reached through sven's `brain` driver, which honours every sampling control
and reads the policy trace.

**Fix:** a model binding may name the agent runtime's driver (`ModelBinding::driver`, absent from every spec written
before it); the worker uses it for the provider, and an online run's rollouts bind the learner's server with `brain`.
Spec: `an_online_rollout_is_given_the_tools_its_task_declares_and_graded_by_its_verifier` (splinter-factory
`it::online::worker`, `make test/online/tools`), and the worker's
`test_a_model_bound_with_a_driver_is_reached_through_that_driver`.
