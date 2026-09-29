#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

# No unreviewed performance numbers in prose or in source narration.
#
# A measured number written into a README, a comment, a doc comment or a
# string literal - "prefill took 72 s", "3.6x faster" - is a claim that
# outlives the hardware, driver and code that produced it; once any of them
# changes, the number is a lie nobody can detect. Measurements live in the
# test that asserts them or in the report a run writes, where they are
# reproduced, not in narration.
#
# Scopes: every README.md (every line), and the comments and string
# literals of crates/, experiments/ and scripts/ (code itself is exempt: a
# `Duration::from_secs(30)` is a value the program depends on, not a claim).
#
# Denied: a number next to a duration (ms, s, min), a rate (fps, tok/s,
# GB/s, FLOP/s) or a percentile with a value; a percentage or an Nx
# multiplier only when measurement vocabulary is nearby, so "30% masking"
# (a hyperparameter) is not confused with "reached 93% of peak" (a result).
# Sizes, dimensions, ports and requirements are not claims.
#
# Escape hatch: `perf-number: <reason>` in a comment on the same line or the
# line before (`<!-- perf-number: ... -->` in Markdown) marks a reviewed,
# deliberate number - a datasheet constant, a fixture whose value IS the
# number, an experiment report's measured result.
#
# scripts/gates/ is exempt: a gate names the patterns it enforces. This gate
# is adapted from the one the brain repository runs.
#
# Usage: scripts/gates/check-no-perf-numbers.sh [file ...]

set -u
cd "$(git rev-parse --show-toplevel)" || exit 1

fail=0
tmp_hits=$(mktemp)
tmp_src=$(mktemp)
tmp_cand=$(mktemp)
tmp_txt=$(mktemp)
tmp_ctx=$(mktemp)
trap 'rm -f "$tmp_hits" "$tmp_hits.sorted" "$tmp_src" "$tmp_cand" "$tmp_txt" "$tmp_ctx"' EXIT

# A number that is not glued to a surrounding word: "0.19", "36", "1,024" -
# but not the "32" of `u32s`, the "40" of `P40s`, or the "4" of `f16x4`.
NUM='(?<![A-Za-z0-9_.])[0-9](?:[0-9.,]*[0-9])?'
# "Hard" units: unambiguous performance vocabulary, no English word ever
# collides with them, so prose needs no context-gating for these.
HARD_RE="$NUM"'[[:space:]]?(ms|min|s\/(page|frame)|fps|tok(en)?s?\/s|[GM]B\/s|[TG]FLOP(S|s)?(\/s)?)\b'
# Bare seconds. `s` is also the stride/index name that WGSL and conv configs
# use, so a trailing `=`/`-` (as in "k=3 s=2", "(k-1,s-1)") is not a duration.
HARD_RE_BARE_S="$NUM"'[[:space:]]?s\b(?![[:space:]]*[=-])'
# p50/p95/p99 only counts once an actual value trails it (not just the
# percentile named in prose, e.g. "latency p50/p99 + throughput").
PCTL_RE='\bp(50|95|99)\b[^0-9a-zA-Z\n]{0,20}[0-9][0-9.,]*[[:space:]]?(ms|s|%)\b'

# "Soft" patterns: a bare percentage or an Nx/Nx multiplier. Both also occur
# as architecture/config constants (a 16x downsample, 30% token masking), so
# a hit only counts if measurement vocabulary shows up within a few lines -
# perf prose in this repo is written in wrapped paragraphs, so the cue word
# ("measured", "speedup", ...) is often a line or two away from the number
# itself, not on the exact same line. A trailing `}` excludes a hex format
# specifier (`{hash:016x}`), which is a format string, not a multiplier.
SOFT_PCT_RE="$NUM"'[[:space:]]?%'
SOFT_X_RE="$NUM"'[xx](?![0-9a-zA-Z}])'
CONTEXT_RE='measur|profil|\bwall\b|laten|throughput|speedup|speed-up|\bfaster\b|\bslower\b|benchmark|median|\bmean\b|\bpeak\b|regress(ed|ion)?|baseline|roofline|\bflop|decode (loop|step)|training step|step time|per-kernel|per-frame|per-page|inference time|resident instance|wall-clock|wall time'
# Code prose reports a measurement with different words than a doc paragraph
# does, and a rate unit anywhere nearby is itself the strongest cue there is.
# An EXTENSION of CONTEXT_RE, never a restatement of it, so the two scopes
# cannot drift into two different ideas of what "talks about measurement".
SRC_CONTEXT_RE="$CONTEXT_RE"'|\btook\b|\btakes\b|\bcosts?\b|\bspent\b|dominat|elapsed|\broof\b|occupancy|bandwidth|[GM]B\/s|tok(en)?s?\/s|\bfps\b|FLOP|s\/(page|frame)|per (forward|step|token|frame|card|call|image|layer|block|batch|chunk|page|row|tile)'

is_escaped() {
  # $1 = file, $2 = 1-based line number: escaped if the marker is on this
  # line or the one immediately before it.
  local f="$1" ln="$2"
  sed -n "$((ln - 1 > 0 ? ln - 1 : 1)),${ln}p" "$f" | grep -q 'perf-number:'
}

has_context() {
  # $1 = file, $2 = 1-based line number: true if measurement vocabulary
  # appears in the surrounding window (4 lines back, 2 forward - covers a
  # wrapped paragraph's cue word and a heading that introduces a table).
  local f="$1" ln="$2" start end
  start=$((ln - 4)); [ "$start" -lt 1 ] && start=1
  end=$((ln + 2))
  sed -n "${start},${end}p" "$f" | grep -qiE "$CONTEXT_RE"
}

# ---------------------------------------------------------------- prose ----
while IFS= read -r -d '' file; do
  # Hard, unambiguous units.
  while IFS=: read -r ln rest; do
    [ -z "$ln" ] && continue
    is_escaped "$file" "$ln" && continue
    printf '%s\n' "$file:$ln: $rest" >> "$tmp_hits"
  done < <(grep -noP "$HARD_RE" "$file")

  # Bare seconds ("72.2 s") - same hard-unit treatment, kept as its own pass
  # only because it's expressed separately above for readability.
  while IFS=: read -r ln rest; do
    [ -z "$ln" ] && continue
    is_escaped "$file" "$ln" && continue
    printf '%s\n' "$file:$ln: $rest" >> "$tmp_hits"
  done < <(grep -noP "$HARD_RE_BARE_S" "$file")

  # p50/p95/p99 followed by an actual value.
  while IFS=: read -r ln rest; do
    [ -z "$ln" ] && continue
    is_escaped "$file" "$ln" && continue
    printf '%s\n' "$file:$ln: $rest" >> "$tmp_hits"
  done < <(grep -noiP "$PCTL_RE" "$file")

  # Soft patterns: percentage and Nx/Nx, gated on nearby measurement context.
  while IFS=: read -r ln rest; do
    [ -z "$ln" ] && continue
    is_escaped "$file" "$ln" && continue
    has_context "$file" "$ln" || continue
    printf '%s\n' "$file:$ln: $rest" >> "$tmp_hits"
  done < <(grep -noP "$SOFT_PCT_RE" "$file")

  while IFS=: read -r ln rest; do
    [ -z "$ln" ] && continue
    is_escaped "$file" "$ln" && continue
    has_context "$file" "$ln" || continue
    printf '%s\n' "$file:$ln: $rest" >> "$tmp_hits"
  done < <(grep -noP "$SOFT_X_RE" "$file")
done < <(
  if [ "$#" -gt 0 ]; then
    for f in "$@"; do case "$f" in *README.md) printf '%s\0' "$f" ;; esac; done
  else
    git ls-files -z '*README.md'
  fi
)

# --------------------------------------------------------------- source ----
# Narration only: every comment body and string literal, one record per
# source line that has any, as `path:lineno:text`. Blanking the code (rather
# than dropping it) keeps column-free line numbers exact, and dropping the
# all-blank records keeps this corpus small enough to scan in one pass.
if [ "$#" -gt 0 ]; then
  src_files=$(printf '%s\n' "$@" | grep -E '^(crates|experiments|scripts)/' |
    grep -E '(\.rs|\.wgsl|\.py|\.sh)$|^scripts/hooks/' | grep -v '^scripts/gates/')
else
  src_files=$(find crates experiments scripts -type f ! -path '*/__pycache__/*' \
    \( -name '*.rs' -o -name '*.wgsl' -o -name '*.py' -o -name '*.sh' \
       -o -path 'scripts/hooks/*' \) -print | grep -v '^scripts/gates/' | sort)
fi
[ -n "$src_files" ] && printf '%s\n' "$src_files" | xargs awk '
FNR == 1 { inblock = 0; hash = (FILENAME ~ /\.(sh|py)$/ || FILENAME ~ /^scripts\/hooks\//) }
{
  line = $0; out = ""; n = length(line); i = 1
  while (i <= n) {
    c = substr(line, i, 1)
    if (inblock) {
      rest = substr(line, i); p = index(rest, "*/")
      if (p > 0) { out = out substr(rest, 1, p - 1); i = i + p + 1; inblock = 0 }
      else { out = out rest; i = n + 1 }
      continue
    }
    two = substr(line, i, 2)
    if (!hash && two == "//") { out = out substr(line, i + 2); break }
    if (!hash && two == "/*") { inblock = 1; i += 2; continue }
    if (hash && c == "#") { out = out substr(line, i + 1); break }
    if (c == "\"" || (hash && c == "'\''")) {
      q = c; i++
      while (i <= n) {
        d = substr(line, i, 1)
        if (d == "\\") { i += 2; continue }
        if (d == q) { i++; break }
        out = out d; i++
      }
      out = out " "; continue
    }
    out = out " "; i++
  }
  gsub(/[[:space:]]+$/, "", out)
  if (out != "") print FILENAME ":" FNR ":" out
}' > "$tmp_src"

# The same records with the `path:lineno:` key stripped, so neither the unit
# patterns nor the context-gate can match a file name or a line number.
sed 's/^[^:]*:[0-9]*://' "$tmp_src" > "$tmp_txt"

# Candidates: record number + matched text, one line per match (every pattern
# is context-gated in this scope, so they need no per-pattern tagging).
: > "$tmp_cand"
for pat in "$HARD_RE" "$HARD_RE_BARE_S" "$SOFT_PCT_RE" "$SOFT_X_RE"; do
  grep -noP "$pat" "$tmp_txt" >> "$tmp_cand"
done
grep -noiP "$PCTL_RE" "$tmp_txt" >> "$tmp_cand"

# Which records carry measurement vocabulary. Done once with grep -E (rather
# than inside the awk below) so SRC_CONTEXT_RE keeps `\b` and stays a plain
# extension of CONTEXT_RE.
grep -niE "$SRC_CONTEXT_RE" "$tmp_txt" | cut -d: -f1 > "$tmp_ctx"

# Context-gate every candidate against the narration window (same file, 4
# source lines back, 2 forward) and report the survivors.
awk -F: -v ctxf="$tmp_ctx" -v candf="$tmp_cand" '
FILENAME == ctxf { hasctx[$1 + 0] = 1; next }
FILENAME == candf {
  rec = $1; sub(/^[0-9]+:/, "", $0); cand[rec] = cand[rec] "\n" $0; next
}
{ path[FNR] = $1; lno[FNR] = $2 + 0 }
END {
  for (r in cand) {
    r += 0
    hit = 0
    # Narration records are contiguous per file but skip code-only lines, so
    # the +-4/+-2 SOURCE-line window is at most 7 records wide; +-8 covers it.
    for (j = r - 8; j <= r + 8; j++) {
      if (!(j in path) || !(j in hasctx)) continue
      if (path[j] != path[r]) continue
      if (lno[j] < lno[r] - 4 || lno[j] > lno[r] + 2) continue
      hit = 1; break
    }
    if (!hit) continue
    n = split(cand[r], parts, "\n")
    for (k = 1; k <= n; k++)
      if (parts[k] != "") print path[r] ":" lno[r] ": " parts[k]
  }
}' "$tmp_ctx" "$tmp_cand" "$tmp_src" | while IFS=: read -r file ln rest; do
  is_escaped "$file" "$ln" && continue
  printf '%s\n' "$file:$ln:$rest" >> "$tmp_hits"
done

if [ -s "$tmp_hits" ]; then
  # Sort by file/line for a stable, readable order. NOT `sort -u` restricted
  # to the file:line keys - two distinct matches (e.g. "1.3x" and "1.6x") can
  # legitimately share one line, and keying uniqueness on file:line alone
  # would silently drop the second one. `uniq` on the fully-sorted whole line
  # instead only collapses genuine exact-duplicate hits (the same match
  # caught by two overlapping patterns).
  sort -t: -k1,1 -k2,2n "$tmp_hits" | uniq > "$tmp_hits.sorted"
  while IFS= read -r hit; do
    echo "PERF NUMBER: $hit"
  done < "$tmp_hits.sorted"
  count=$(wc -l < "$tmp_hits.sorted")
  fail=1
fi

if [ "$fail" -ne 0 ]; then
  echo
  echo "Found $count bare perf number(s) (see PERF NUMBER lines above,"
  echo "file:line). Each one is a wall-clock/throughput/percentage claim"
  echo "written into prose or into a comment/doc-comment/string literal"
  echo "without a reviewed exception. Fix each by either:"
  echo "  1. Rephrasing to drop the specific bare number - keep the claim and"
  echo "     its reasoning, and point at what reproduces it (the experiment or test,"
  echo "     or the report a run writes) instead of a fixed"
  echo "     figure, or"
  echo "  2. Marking it a deliberate, reviewed exception with"
  echo "     'perf-number: <reason>' on the same line or the line immediately"
  echo "     before it ('<!-- perf-number: ... -->' in markdown,"
  echo "     '// perf-number: ...' / '# perf-number: ...' in source)."
  exit 1
fi
echo "check-no-perf-numbers: OK"
