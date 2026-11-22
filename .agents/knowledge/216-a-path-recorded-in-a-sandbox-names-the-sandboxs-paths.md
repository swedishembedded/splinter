<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 216. A path recorded in a sandbox names the sandbox's paths; the host fork must take them as its own tree

**Lesson:** a trial's agent works at the path the sandbox shows the tree at (`/workspace/repo`), and a capable agent
names it in absolute form (`workdir`, `read_file` paths). The repair fork runs the engine on the host, rooted at a host
copy of the tree, so every such call was refused as outside the root and no path of that agent replayed. The fork now
takes the sandbox's path as another name of its root: the practice session announces `sandbox_workspace`, splinter
passes it to the fork (`ForkAt::workspace_seen_as`), and sven maps it (`EngineBuilder::project_root_seen_as`, also in
the fork's replay tools), still held to the root. Specs: sven `crates/sdk/tests/it/fork.rs`,
`crates/sdk/tests/project_root.rs`, `crates/tool-api/src/path_scope.rs`; splinter
`crates/pipelines/tests/it/path_guided.rs::a_path_naming_the_sandboxs_workspace_replays_on_the_host_tree`.

## Evidence (2026-11-22, lane PLAY, E4 on support)

Before the fix, 992 of 992 path-guided attempts on the base 4B's support failures were refused with `path
'/workspace/repo' is outside the project root` (the 27B teacher names it in almost every shell call). After it, 9 of
31 routed tasks got a verified unhinted suffix; the remaining refusals are differences between the trial's and the
session's observations (bench GL-B3), not paths.
