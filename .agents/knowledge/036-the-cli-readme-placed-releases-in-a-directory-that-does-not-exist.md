# 036. The CLI README placed releases in a directory that does not exist

The splinter CLI README said a release is written under
`<state>/releases/<hex>/` with its adapter file and a `manifest.json`, and
that aliases live in `<state>/releases/aliases/<name>`, and its serve example
passed `<state>/releases/<hex>/adapter.safetensors` to `brain serve`. No code
creates that tree. Found while checking every statement of the flywheel paper
against the source: `crates/orchestrator/src/releases.rs` stores the manifest
in the experience database, the adapter stays in the content-addressed
artifact store (`<state>/artifacts/<xx>/<hex>.<ext>`, see
`crates/store/src/artifacts.rs`), an alias is a pointer in the database, and
the only per-alias file is the serving manifest `<state>/serving/<alias>.json`
(`crates/store/src/lib.rs`), which is what `brain serve --adapter-manifest`
follows.

The README now describes that layout and the serve example uses the serving
manifest. How to avoid it: a layout stated in prose is a claim; the paper's
ledger checks the path constants in the code (STRUCTURE claims X05 and X08),
and a state layout described anywhere should name the function that builds it.
