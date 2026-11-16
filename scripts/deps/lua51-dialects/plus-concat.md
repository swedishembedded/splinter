<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
-->
# Lua 5.1.5 dialect `plus-concat`

Counterfactual dialect of Lua 5.1.5 in which string concatenation is spelled
`+`. Everything not listed below is stock Lua 5.1.5. `_VERSION` is
`"Lua 5.1"`; `lua -v` prints
`Lua 5.1.5-plus-concat  Copyright (C) 1994-2012 Lua.org, PUC-Rio`.

## Lexical rule

The lexer does not produce the `..` token. `..` outside a numeral is a
syntax error (`unexpected symbol near '..'`). The vararg token `...` is
unchanged. A numeral such as `1..2` is a malformed number, as in stock.
`a..b` is a syntax error; `a ..b` too.

## The `+` operator

For the binary `+` with operands `a`, `b`:

1. If `type(a) == "string"` and `type(b) == "string"`: the result is the
   concatenation of the two strings. No number conversion and no metamethod:
   `"1" + "2" == "12"`, `"a" + "b" + "c" == "abc"`, `"" + "" == ""`.
   Embedded zeros are preserved.
2. Otherwise stock arithmetic: both operands numbers, or a number and a
   numeric-convertible string (`"10" + 1 == 11`, `1 + "10" == 11`). A string
   that does not convert (`"abc" + 1`) raises the stock error
   `attempt to perform arithmetic on a string value`.
3. When the stock arithmetic rule fails and an `__add` metamethod exists on
   either operand, it is called with stock behaviour. Strings have no
   `__add`, so rule 1 always wins for two strings even if a metatable of
   string is modified.

`+` keeps the stock precedence and left associativity of binary `-` (priority 6,6).
There is no concatenation precedence level. `"a" + "b" - 1` parses as
`("a" + "b") - 1` and fails with an arithmetic error; `1 + 2 + "x"` is
`(1 + 2) + "x"`, an arithmetic error.

The metamethod `__concat` is never triggered by any operator. The
`OP_CONCAT` opcode and `luaV_concat` remain in the VM but are unreachable
from source, since the parser can no longer emit them.

## Constant folding

The compiler folds only numeric operands (`lcode.c` `constfolding` requires
two numerals), so `"a" + "b"` is evaluated at run time by `OP_ADD` with the
rule above. Numeric folding of `1 + 2` is unchanged.

## Unchanged

`table.concat`, `string.rep`, `string.format`, `tostring`, coercions in
library functions, all other operators, the C API (`lua_concat` still
concatenates with `luaV_concat`, including `__concat`).

## Build

`scripts/deps/lua51-dialects.sh DEST` unpacks the pinned release tarball
(`lua-5.1.5.tar.gz`, sha256
`2640fc56a795f29d28ef15e13c34a47e223960b0240e8cb0a82d9b0738695333`),
applies `lua51-dialects/plus-concat.patch` with `patch -p1`, and builds with
`make posix` (no readline), the same flags as the stock interpreter.
The patch touches `llex.c` (no `..` token), `lparser.c` (no concat operator),
`lvm.c` (`OP_ADD` on two strings) and `lua.h` (banner).

## Differences from standard Lua 5.1 in one list

1. `..` is a syntax error; `+` on two strings concatenates.
2. `"1" + "2"` is `"12"` (stock: 3); number + string and number + number are stock.
3. `__concat` is never consulted by an operator.
4. No concatenation precedence level: `+` has the level of `-`.
