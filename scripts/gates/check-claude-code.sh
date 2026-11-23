#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements solutions for running third-party coding agents
# against local models without exposing their operators' accounts for its
# clients. If your team needs expertise in confining agent runtimes, you can
# procure our services by sending an email to info@swedishembedded.com.

# Claude Code starts only through its runner (crates/agent/src/runner/claude_code.rs,
# whose launch rules confine it and let only a local brain server answer it) or the
# execution worker's adapter (workers/harbor/.../agents/claude_code.py: it runs in the
# trial's sandbox, and the one key it hands Claude Code is the trial's gateway token;
# the gateway holds the upstream credential and the worker refuses a literal one), and
# nothing hands it an account. A tracked file fails this gate when it
#
#   1. starts Claude Code another way: a claude command line, "claude" as a
#      program argument, a package-manager run or global install of the CLI,
#      or a path to an interactive install or to an account's login;
#   2. names an Anthropic or Claude account credential outside tests;
#   3. defaults ANTHROPIC_BASE_URL to Anthropic or OpenRouter;
#   4. is a pipeline stage that names the runtime: no stage interprets an
#      episode by which runtime produced it.
#
# The runner and this gate name these things to refuse them.
#
# Usage: scripts/gates/check-claude-code.sh [file ...]
#   No arguments scans every tracked file; with arguments, only those.
set -uo pipefail
cd "$(git rev-parse --show-toplevel)" || exit 1

RUNNER=crates/agent/src/runner/claude_code.rs
ADAPTER=workers/harbor/splinter_harbor_worker/agents/claude_code.py
SELF=scripts/gates/check-claude-code.sh

INVOCATION='(^|[[:space:];|&(`"'"'"'=])claude[[:space:]]+(-p\b|-c\b|--print|--bare|--dangerously|--permission-mode|--output-format|--model|--resume|--continue|--version|--settings|update\b|install\b|mcp\b|config\b)|"claude"[[:space:]]*[],)]|\b(npx|bunx)\b.*@anthropic-ai/claude-code|\b(npm|pnpm|yarn|bun)\b.*[[:space:]](-g|--global)\b.*@anthropic-ai/claude-code|\.local/bin/claude\b|(~|\$\{?HOME\}?|/[[:alpha:]]+/[^/[:space:]]+)/\.claude(\.json|/|\b)'
CREDENTIAL='\b(ANTHROPIC_API_KEY|ANTHROPIC_AUTH_TOKEN|ANTHROPIC_IDENTITY_TOKEN|ANTHROPIC_FOUNDRY_API_KEY|ANTHROPIC_FOUNDRY_AUTH_TOKEN|ANTHROPIC_AWS_API_KEY|CLAUDE_CODE_OAUTH_TOKEN|CLAUDE_CODE_OAUTH_REFRESH_TOKEN|AWS_BEARER_TOKEN_BEDROCK)\b'
HOSTED='ANTHROPIC_BASE_URL[^[:alnum:]]{0,4}[=:][^[:alnum:]]{0,4}https?://[^[:space:]"'"'"']*(anthropic\.com|openrouter\.ai|claude\.ai)'
BRANCHING='claude[-_]?code|ClaudeCode'

# The gate's own specification, run before every full scan: each way around the
# runner fails, a test may name a credential, and ordinary text passes.
self_test() {
    local dir probe expect
    dir=$(mktemp -d) || return 1
    mkdir -p "$dir/tests" "$dir/crates/pipelines/src"
    while IFS='|' read -r expect probe text; do
        printf '%s\n' "$text" > "$dir/$probe"
        "$0" "$dir/$probe" >/dev/null 2>&1
        if [ "$?" -ne "$expect" ]; then
            echo "check-claude-code: self-test: expected exit $expect for: $text"
            rm -rf "$dir"
            return 1
        fi
        rm -f "$dir/$probe"
    done <<'SPEC'
1|a.sh|claude -p 'do it'
1|a.sh|exec claude --dangerously-skip-permissions
1|a.rs|Command::new("claude")
1|a.sh|npx @anthropic-ai/claude-code -p hi
1|a.sh|npm install --global @anthropic-ai/claude-code@1
1|a.sh|~/.local/bin/claude --version
1|a.rs|let login = "$HOME/.claude/.credentials.json";
1|a.rs|env.insert("ANTHROPIC_AUTH_TOKEN", key);
1|a.env|ANTHROPIC_BASE_URL=https://openrouter.ai/api/v1
1|crates/pipelines/src/a.rs|if runtime == "claude-code" { return; }
0|tests/a.rs|std::env::set_var("ANTHROPIC_API_KEY", "x");
0|a.md|Claude Code runs only through its runner.
SPEC
    rm -rf "$dir"
}

if [ "$#" -gt 0 ]; then
    files=("$@")
else
    self_test || exit 1
    mapfile -d '' -t files < <(git ls-files -z)
fi

found=0
report() {
    echo "check-claude-code: $1"
    found=1
}
for f in "${files[@]}"; do
    [ -f "$f" ] || continue
    case "$f" in "$RUNNER"|"$ADAPTER"|"$SELF") continue ;; esac
    grep -Iq . "$f" 2>/dev/null || continue
    while IFS= read -r hit; do
        report "$f:$hit: starts Claude Code outside $RUNNER and $ADAPTER"
    done < <(grep -nE "$INVOCATION" "$f" | cut -d: -f1)
    case "$f" in */tests/*|tests/*|*/tests) ;; *)
        while IFS= read -r hit; do
            report "$f:$hit: names an Anthropic or Claude account credential; no run carries one"
        done < <(grep -nE "$CREDENTIAL" "$f" | cut -d: -f1) ;;
    esac
    while IFS= read -r hit; do
        report "$f:$hit: ANTHROPIC_BASE_URL defaults to a hosted API"
    done < <(grep -nE "$HOSTED" "$f" | cut -d: -f1)
    case "$f" in *crates/pipelines/*|*crates/orchestrator/*|*crates/core/*|*crates/store/*|*crates/expdb/*)
        while IFS= read -r hit; do
            report "$f:$hit: a stage names the agent runtime; episodes are interpreted the same whichever runtime produced them"
        done < <(grep -nE "$BRANCHING" "$f" | cut -d: -f1) ;;
    esac
done
if [ "$found" -eq 0 ]; then
    [ "$#" -eq 0 ] && echo "check-claude-code: OK"
    exit 0
fi
exit 1
