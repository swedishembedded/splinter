<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 072. `lua` on the path was whatever the host's alternatives named, and the runtime accepted it

The three Lua task families (caldate, graphkit, tokenbucket) are programs and tests
written for Lua 5.1, and their tests start a bare `lua`. The `lua` runtime of the
registry was `lua` found on the sandbox's search path, and its version probe only
had to print a version: any interpreter passed. On the host this was written on,
`/usr/bin/lua` is a symbolic link to `/etc/alternatives/lua-interpreter`, so which
Lua ran every Lua family's tests, every Lua task the model generator wrote and every
verifier run depended on a setting nobody recorded; the host reported Lua 5.1.5 on
the day it was read, and changing the alternative to 5.3 or 5.4 would have changed
the meaning of `unpack`, `setfenv`, integer division and `loadstring` under the same
task digests. A task's environment record held the version string and the
executable's digest, so a run could be told apart afterwards, but nothing refused
the wrong one before it started.

Fixes: the registry's `lua` now carries the version it is registered for
(`RuntimeSpec::version_prefix`, `Lua 5.1`) and resolving it refuses an interpreter
whose version line does not begin so, naming what it reported; `make deps/lua`
builds the interpreter from the release tarball after checking the tarball's SHA-256
against the one recorded in `scripts/deps/lua51.sh`, without readline, and
`--runtime-dir DIR/bin` puts it first on the search path of the runtime and of the
workspace commands (the confinement reads the directory from that path, so no other
configuration is needed); `scripts/tasks/families.py check` refuses a family run
whose `lua` is not 5.1. Spec: `crates/sandbox/tests/runtime_registry.rs`
(`the_lua_runtime_is_lua_5_1_and_an_interpreter_that_reports_another_version_is_refused`).

Not covered: the families' own test hosts start `lua` by name from their `PATH`; a
run under `--runtime-dir` finds the pinned one first, a run without it finds the
system's, which the family check and the runtime refuse unless it is 5.1.
