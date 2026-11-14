<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 153. A confinement whose network was all or nothing could not hold an agent to its one model server

**Lesson:** an agent runtime that is a separate binary talks to its model over TCP, so a confinement that offers only
"no network" or "the whole network" either breaks it or lets it reach any host, the operator's hosted accounts
included. Confine it to the one port its model server listens on, and close every other socket family.

## Evidence

Claude Code as a runtime behind the agent-runner contract (`crates/agent/src/runner`, 2026-11-15): it must reach a
local brain server, and nothing else, because a Claude Code process holds the operator's login when it can read the
account's home and calls `api.anthropic.com` directly for some checks whatever its base URL says. `Confinement` gained
`tcp_connect_ports`: Landlock (ABI 4) port rules allow TCP connect to those ports, TCP bind stays denied, and a seccomp
filter leaves only IPv4 stream sockets (no UDP, no IPv6, no packet sockets). Gated by
`a_confinement_with_tcp_ports_reaches_only_those_ports` (`crates/sandbox/tests/it/confinement.rs`: the allowed port
connects, another is "Permission denied", UDP and IPv6 cannot be made) and by
`the_confined_run_cannot_read_the_login_nor_reach_another_port` (`crates/agent/tests/it/claude_code.rs`: a stand-in for
the binary, pinned by its digest, cannot read a canary in a stand-in login by absolute path, reaches the endpoint's
port and not another, and holds the server's key as its only credential).

## Not done

The runner has run only stand-in binaries in tests here; the pinned Claude Code build has run against brain through
bench's launcher, which applies the same rules.
