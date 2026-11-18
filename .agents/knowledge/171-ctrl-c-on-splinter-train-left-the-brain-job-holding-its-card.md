<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 171. Ctrl-C on splinter train left the brain job holding its card

**Lesson:** a served job is detached from its client by design, so a client that dies on a signal
leaves its job running. The client must turn the signal into the cancel it already sends for a
`runs cancel` request, and say whether the service confirmed it.

Found 2026-11-18 in pilot A2 (splinter 3fc94481f, brain c25d5c2f4): SIGINT to `splinter train`
killed the process at once; the LoRA job kept training on its GH200, and with no job listing
(`GET /v1/jobs` was 405 then) its id was lost, so the only way to free the card was to restart the
server. The served job client already cancelled on its token (`DELETE /v1/jobs/{id}`); nothing fired
the token on a signal. Since then every recorded run watches `splinter_orchestrator::interrupt`
(`tests/interrupt.rs` sends a real SIGINT), and the cancel's outcome is in the error
(`served_train::cancelling_stops_the_job_on_the_service`, against the scripted brain double).
