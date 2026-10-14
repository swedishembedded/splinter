# 026. A test that returns early reports a pass

Five tests printed "SKIPPED" and returned because git, a docker daemon or a
non-root user was missing. The harness reported them as passed. A suite that
is green because its checks did not run is a false success, and the same shape
let a brain test that never ran its body read as "1 passed" in 0.00 s.

How to avoid it: a test that cannot run fails with the reason, or carries
#[ignore = "reason"] so every run shows it as ignored. The gate
check-test-skips refuses the print-and-return shape. After writing a test that
needs a fixture, run it once where the fixture is missing and see that it does
not pass.
