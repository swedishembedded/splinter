<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
-->
# Lua 5.1.5 dialect `zero-index`

Counterfactual dialect of Lua 5.1.5 in which string positions start at 0.
Everything not listed below is stock Lua 5.1.5. `_VERSION` is `"Lua 5.1"`;
`lua -v` prints `Lua 5.1.5-zero-index  Copyright (C) 1994-2012 Lua.org, PUC-Rio`.

## Position conversion

Every position argument `p` of the affected functions is converted to a stock
1-based position `q` before the stock algorithm runs:

- `p >= 0`: `q = p + 1`
- `p < 0`: `q = p` (stock negative rule, so `-1` is the last character;
  stock clamping applies afterwards: `q = len + 1 + p`, and below 1 it is 1).

Results that are positions are converted back by subtracting 1.

## Functions

- `string.sub(s, i [, j])`: `i` and `j` converted as above; default `j` is `-1`.
  `i` is required, as in stock. Stock clamping: start below the first
  character becomes the first, end past the last becomes the last, start
  after end gives `""`. `("hello"):sub(0,1) == "he"`, `sub(5) == ""`,
  `sub(-3) == "llo"`, `sub(0,-2) == "hell"`.
- `string.byte(s [, i [, j]])`: default `i` is `0`; default `j` is the
  converted `i` (an explicit `j` is converted). Empty or out-of-range
  interval returns no values.
- `string.find(s, pattern [, init [, plain]])`: `init` converted (default
  `0`; an `init` past the end is clamped to the end of the string, as in
  stock). Returns 0-based start and inclusive 0-based end (stock result
  minus 1 each). An empty match at offset `p` returns `p, p-1`
  (`("hello"):find("") -> 0 -1`). Capture results follow the start and end.
- `string.match(s, pattern [, init])`: `init` as for `find`.
- Position captures `()` in `find`, `match`, `gmatch` and `gsub` (including
  the `%n` replacement form and table/function replacements) yield the
  0-based offset: `("flaaap"):match("()aa()") -> 2 4`; at the end of the
  subject the capture is `string.len(s)`.
- `string.gmatch` and `string.gsub` take no position arguments.
- Operator `#` on a string yields `string.len(s) - 1` (`#"abc" == 2`,
  `#"" == -1`). `#` is not constant-folded by the compiler (stock does not
  fold `OP_LEN` either), so literals behave the same as variables.

## Unchanged

`#` on tables and the `__len` metamethod on non-strings, `string.len`,
`string.rep`, `string.upper`, `string.lower`, `string.reverse`,
`string.format`, `string.char`, `string.dump`, all of `table`, `math`, `io`,
`os`, table indices (still from 1), the C API (`lua_objlen`, `luaL_*`).

## Build

`scripts/deps/lua51-dialects.sh DEST` unpacks the pinned release tarball
(`lua-5.1.5.tar.gz`, sha256
`2640fc56a795f29d28ef15e13c34a47e223960b0240e8cb0a82d9b0738695333`),
applies `lua51-dialects/zero-index.patch` with `patch -p1`, and builds with
`make posix` (no readline), the same flags as the stock interpreter.
The patch touches `lstrlib.c` (positions), `lvm.c` (`OP_LEN` on strings) and
`lua.h` (banner).

## Differences from standard Lua 5.1 in one list

1. `sub`/`byte`/`find`/`match` take 0-based non-negative positions.
2. `find` returns 0-based start and end.
3. `()` captures are 0-based.
4. `#s` is `string.len(s) - 1` for strings.
5. Default `byte` start is `0`; default `find`/`match` `init` is `0`.
