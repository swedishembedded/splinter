<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 140. A reference must solve a reset workspace again, not run twice on its own result

The first robustness gate required that running the reference solution a second time on the state it had just produced still graded solved. It rejected stateful environments for a reason that is not a defect: at the lowest difficulty of the database workflow (one request, no traps) the request is "restock this product to at least N units", whose reference adds the missing units, and adding them again is a different state. One of four environments of the first compile was rejected for it ("running the reference a second time on its own result breaks it"). The gate now asks what the environment promises: after the environment's own reset, the workspace grades unsolved and the reference solves it again.

Rule: test an environment's reference against the environment's reset, never against its own output; idempotence is a property of the reset, not of the task. Evidence: `crates/knowledge/src/factory/admission.rs` (`check_robustness`), the compile test in `crates/knowledge/tests/it/factory_store.rs`.
